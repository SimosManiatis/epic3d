import open3d as o3d
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom  # Import zoom function from scipy
import torch.optim as optim #for the Adam
import time #for the training loop as a time limit
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import visdom #visualization
#from torch.optim.lr_scheduler import ReduceLROnPlateau

# Setting up TensorBoard
log_dir = r"D:\CI\Python Files\try2\TensorBoard"
writer = SummaryWriter(log_dir)
viz = visdom.Visdom() #start up a new session to visualize the eval phase of the training loops
model_save_dir = r"D:\CI\Python Files\try2\aaaaaaaaaaaa"
#set device, be careful to detach later when necesary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#choose folder where all the objs are located
#directory_path = r"D:\CI\Python Files\try2\new\corrected"
directory_path = r"D:\CI\Python Files\try2\new\corrected"

#Choose initial voxel size for conversion from obj to voxel grids
voxel_size = 0.1

#Not Desirable, but since there are many errors and mismatches when the tensors are not uniformed, 
#we set a target scale (cube-like)
target_shape = (32, 32, 32)  # Define target shape for all tensors

#First check if the file directory has obj files inside, then if those 3D objects have triangles,
#we can procceed or else we ignore them.
def load_and_check_meshes(directory_path):
    files_in_directory = os.listdir(directory_path)
    obj_files = [file for file in files_in_directory if file.endswith('.obj')]
    meshes_with_triangles = []
    for obj_file in obj_files:
        full_path = os.path.join(directory_path, obj_file)
        mesh = o3d.io.read_triangle_mesh(full_path)
        if not mesh.is_empty() and len(mesh.triangles) > 0:
            meshes_with_triangles.append(mesh)
        else:
            print(f"{obj_file} does not contain triangles.")
    return meshes_with_triangles

#since sometimes the 3d files may have various sources use this definition to center all at the same origin
def center_mesh(mesh):
    centroid = mesh.get_center()
    mesh.translate(-centroid)
    return mesh

#depending on the voxel size
def mesh_to_voxel(mesh, voxel_size):
    return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

#we convert to pytorch tensors to use later for training the GAN
def voxel_grid_to_tensor(voxel_grid, target_shape):
    voxels = np.asarray(voxel_grid.get_voxels())
    if len(voxels) == 0:
        return torch.empty(0)
    voxel_positions = np.array([voxel.grid_index for voxel in voxels])
    if voxel_positions.size == 0:
        return torch.empty(0)
    max_coords = voxel_positions.max(axis=0)
    tensor_shape = tuple(max_coords + 1)
    voxel_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
    voxel_tensor[tuple(voxel_positions.T)] = 1
    
    # Rescale tensor to target shape
    scaling_factors = [target_shape[i] / voxel_tensor.shape[i] for i in range(3)]
    voxel_tensor_rescaled = torch.tensor(zoom(voxel_tensor, scaling_factors, order=0))  # Use nearest neighbor scaling
    
    # Print the final shape of the tensor
    print("Final tensor shape:", voxel_tensor_rescaled.shape)
    
    return voxel_tensor_rescaled


#the following definitions are to check if the conversion from obj to voxel grid and then to tensor went well
#added an aspect ratio fix cause sometimes the bounding box was off which is we weird
def visualize_tensor(tensor):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    tensor_np = tensor.cpu().numpy()
    filled_voxels = np.argwhere(tensor_np > 0)
    ax.scatter(filled_voxels[:, 0], filled_voxels[:, 1], filled_voxels[:, 2])
    max_range = np.array([filled_voxels[:, 0].max()-filled_voxels[:, 0].min(), 
                          filled_voxels[:, 1].max()-filled_voxels[:, 1].min(), 
                          filled_voxels[:, 2].max()-filled_voxels[:, 2].min()]).max() / 2.0
    mid_x = (filled_voxels[:, 0].max() + filled_voxels[:, 0].min()) * 0.5
    mid_y = (filled_voxels[:, 1].max() + filled_voxels[:, 1].min()) * 0.5
    mid_z = (filled_voxels[:, 2].max() + filled_voxels[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

# Main process
meshes = load_and_check_meshes(directory_path)
centered_meshes = [center_mesh(mesh) for mesh in meshes]
voxel_grids = [mesh_to_voxel(mesh, voxel_size) for mesh in centered_meshes]
voxel_tensors = [voxel_grid_to_tensor(voxel_grid, target_shape) for voxel_grid in voxel_grids]

#give the user the option, to choose one of the objects for evaluation
print("Hey! Choose one of the models for evaluation.")
for i, _ in enumerate(centered_meshes):
    print(f"{i}: Model {i + 1}")

try:
    model_index = int(input("Enter the model number: "))
    if 0 <= model_index < len(centered_meshes):
        print("Visualizing the voxel grid:")
        o3d.visualization.draw_geometries([voxel_grids[model_index]])
        print("Visualizing the tensor representation:")
        visualize_tensor(voxel_tensors[model_index])
    else:
        print("Invalid model number.")
except ValueError:
    print("Please enter a valid integer.")




#Generator Number of layers=5
class net_G(torch.nn.Module):
    def __init__(self, z_dim, cube_len, bias):
        super(net_G, self).__init__()
        # No need for self.args = args
        self.cube_len = cube_len
        self.bias = bias
        self.z_dim = z_dim
        self.f_dim = 32

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = self.conv_layer(self.z_dim, self.f_dim*8, kernel_size=4, stride=2, padding=padd, bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim*8, self.f_dim*4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*4, self.f_dim*2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*2, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def forward(self, x):
        out = x.view(-1, self.z_dim, 1, 1, 1)
        # print(out.size())  # torch.Size([32, 200, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = torch.squeeze(out)
        return out


#Critic Number of Layers=5
class net_D(torch.nn.Module):
    def __init__(self, cube_len, leak_value, bias):
        super(net_D, self).__init__()
        # No need for self.args = args
        self.cube_len = cube_len
        self.leak_value = leak_value
        self.bias = bias
        self.f_dim = 32

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        # print(out.size()) # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        # out = out.view(-1, 256*2*2*2)
        # print (out.size())
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out
    


#Hyperparameters
epochs = 2000
batch_size = 4
d_lr = 0.00005
g_lr = 0.00035
beta1 = 0.5
beta2 = 0.999
z_dim = 50
cube_len = 32
leak_value = 0.2
n_critic = 1
bias = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dis = "norm"



# Instantiate generator and discriminator
generator = net_G(z_dim=z_dim, cube_len=cube_len, bias=bias).to(device)
discriminator = net_D(cube_len=cube_len, leak_value=0.2, bias=bias).to(device)
voxel_tensors = [voxel_grid_to_tensor(voxel_grid, target_shape) for voxel_grid in voxel_grids]
tensor_dataset = torch.utils.data.TensorDataset(torch.stack(voxel_tensors))
total_length = len(tensor_dataset)
train_length = int(total_length * 0.8)
validation_length = total_length - train_length
train_dataset, validation_dataset = torch.utils.data.random_split(tensor_dataset, [train_length, validation_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1,beta2))
g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1,beta2))
criterion_D = torch.nn.MSELoss()  # Critic loss
criterion_G = torch.nn.L1Loss()  # Generator loss
# Add model to TensorBoard
sample_z = torch.randn(1, z_dim, 1, 1, 1, device=device)
sample_voxel = torch.randn(1, 1, cube_len, cube_len, cube_len, device=device)
writer.add_graph(generator, sample_z)
writer.add_graph(discriminator, sample_voxel)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def generateZ(args,batch):

    if   z_dis == "norm":
         Z = torch.Tensor(batch, z_dim).normal_(0, 0.33).to(device)
    elif z_dis == "uni":
         Z = torch.randn(batch, z_dim).to(device)
    else:
        print("z_dist is not normal or uniform")

    return Z






def visualize_voxels(voxels):
    # voxels is a binary numpy array of shape (D, H, W)
    voxels = np.pad(voxels, 1, 'constant', constant_values=False)  # Add padding to ensure edges are visible
    x, y, z = voxels.nonzero()
    points = np.column_stack((x, y, z))
    viz.scatter(
        X=points,
        opts=dict(
            title="Current Generator Output",
            markersize=5,
            markercolor=np.array([[0, 255, 255]]),  # red color
            xtickmin=0,
            xtickmax=voxels.shape[0],
            ytickmin=0,
            ytickmax=voxels.shape[1],
            ztickmin=0,
            ztickmax=voxels.shape[2],
           three_d=True
        )
    )

# Function to visualize generated voxels using Visdom
def save_generated_samples(epoch, generator, fixed_noise):
    print(f"Generating samples for epoch {epoch}")
    with torch.no_grad():
        generated_voxels = generator(fixed_noise).detach().cpu()
    voxel_array = generated_voxels[0].numpy() > 0.1
    visualize_voxels(voxel_array)
    print("Sample visualization complete")
    


def voxel_to_point_cloud(voxel_grid):
    """
    Converts a voxel grid to a point cloud.
    Args:
    - voxel_grid (numpy.ndarray): The voxel grid as a 3D NumPy array.
    Returns:
    - o3d.geometry.PointCloud: Open3D point cloud object.
    """
    points = np.argwhere(voxel_grid)  # Extract the coordinates of active voxels
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return point_cloud

def save_point_cloud_as_ply(point_cloud, filename):
    """
    Saves a point cloud to a PLY file.
    Args:
    - point_cloud (o3d.geometry.PointCloud): The point cloud object.
    - filename (str): Full path to save the PLY file.
    """
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)
    print(f"Saved point cloud to {filename}")
    



# Fixed noise for watching the progress of generated samples
fixed_noise = generateZ({'z_dim': z_dim}, batch_size)
def train_gan(generator, discriminator, train_loader, validation_loader, fixed_noise, n_critic=1):
    logger.info("Starting GAN Training")

    # Track start time
    start_time = time.time()

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                generator.train()
                discriminator.train()
            else:
                generator.eval()
                discriminator.eval()
            data_loader = train_loader if phase == 'train' else validation_loader

            running_loss_G = 0.0
            running_loss_D = 0.0
            total_batches = 0

            for voxel_tensors in tqdm(data_loader):
                voxel_tensors = voxel_tensors[0].to(device)
                batch_size = voxel_tensors.size(0)

                noise = generateZ(z_dim, batch_size).to(device)

                ### Discriminator Training
                d_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # Train with real images
                real_output = discriminator(voxel_tensors)
                d_loss_real = criterion_D(real_output, real_labels)
                
                # Generate fake images and train
                fake_voxels = generator(noise)
                fake_output = discriminator(fake_voxels.detach())
                d_loss_fake = criterion_D(fake_output, fake_labels)

                # Combine losses and update discriminator
                d_loss = d_loss_real + d_loss_fake
                if phase == 'train':
                    d_loss.backward()
                    d_optimizer.step()

                ### Generator Training
                # Only update generator once per n_critic iterations of discriminator
                if total_batches % n_critic == 0:
                    g_optimizer.zero_grad()
                    # It's important to calculate fake output again since the discriminator is updated
                    fake_output = discriminator(fake_voxels)
                    g_loss = criterion_G(fake_output, real_labels)
                    if phase == 'train':
                        g_loss.backward()
                        g_optimizer.step()

                running_loss_G += g_loss.item() * n_critic  # Adjusted for n_critic scaling
                running_loss_D += d_loss.item()
                total_batches += 1

            epoch_loss_G = running_loss_G / total_batches
            epoch_loss_D = running_loss_D / total_batches

            logger.info(f"Epoch {epoch+1}/{epochs}, Phase: {phase}, Loss G: {epoch_loss_G:.4f}, Loss D: {epoch_loss_D:.4f}")
            if phase == 'train':
                writer.add_scalar('Loss/Train/Generator', epoch_loss_G, epoch)
                writer.add_scalar('Loss/Train/Discriminator', epoch_loss_D, epoch)
            else:
                writer.add_scalar('Loss/Val/Generator', epoch_loss_G, epoch)
                writer.add_scalar('Loss/Val/Discriminator', epoch_loss_D, epoch)

        # Outside of data loader loop - Check epoch for saving generated samples
        if (epoch + 1) % 200 == 0:  # Adjust as needed
            save_generated_samples(epoch + 1, generator, fixed_noise)
            with torch.no_grad():
                generator.eval()
                fixed_output = generator(fixed_noise)  # Assuming fixed_noise is a batch of latent vectors
                voxel_grid = fixed_output.cpu().numpy() > 0.5  # Threshold to create a binary voxel grid
                point_cloud = voxel_to_point_cloud(voxel_grid[0])  # Convert first item in batch
                save_point_cloud_as_ply(point_cloud, f'{model_save_dir}/generated_epoch_{epoch+1}.ply')
                # Update learning rates at the end of each epoch
                #d_scheduler.step(epoch_loss_D)
                #g_scheduler.step(epoch_loss_G)

    # Save final model weights after training completes
    torch.save(generator.state_dict(), os.path.join(log_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(log_dir, 'discriminator_final.pth'))
    writer.close()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed in {total_time // 60}m {total_time % 60}s")


# Start training
train_gan(generator, discriminator, train_loader, validation_loader, fixed_noise)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 50  # Assuming z_dim is 50 as per your setup
generator = net_G(z_dim=z_dim, cube_len=32, bias=False).to(device)

# Correct path to the model file
model_path = r'D:\CI\Python Files\try2\logs\generator_final.pth'

# Load the trained generator model
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Visualization and noise generation functions
def generate_unique_z(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1, 1, 1).to(device)

def visualize_voxels_viz(voxel_tensor, win_title="Voxel Model"):
    voxel_array = voxel_tensor.squeeze().cpu().numpy() > 0.05
    z, y, x = np.indices(voxel_array.shape).astype(float) + 0.5
    x, y, z = x[voxel_array], y[voxel_array], z[voxel_array]
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    
    viz = visdom.Visdom()
    viz.scatter(
        X=points,
        opts=dict(
            title=win_title,
            markersize=2,
            markercolor=np.array([[0, 255, 0]]),  # RGB color
            webgl=True,
        )
    )

# Generate and visualize 4 new models with unique Z values using Visdom
for i in range(4):
    with torch.no_grad():
        z = generate_unique_z(1, z_dim, device)
        generated_voxels = generator(z)
        visualize_voxels_viz(generated_voxels, win_title=f"Generated Model {i+1}")
