import os
import io
import numpy as np
import json

from pyxtal import pyxtal
from scipy.fft import fft, fft2, fftshift
from scipy.ndimage import shift
from skimage import filters

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from ase.visualize.plot import plot_atoms

# "name" = wallpaper group name, 
# "layer_no" = 3D extension (layer group number), 
# "no_atoms" = bounds for number of atoms in UC, 
# "even" = shoud number of atoms in UC be even?
wallpaper_groups = {1:  {"name" : "p1",   "layer_no" : 1,  "no_atoms" : [1, 10], "even" : False}, 
                    2:  {"name" : "p2",   "layer_no" : 3,  "no_atoms" : [3, 10], "even" : False},
                    3:  {"name" : "pm",   "layer_no" : 11, "no_atoms" : [3, 10], "even" : False},
                    4:  {"name" : "pg",   "layer_no" : 12, "no_atoms" : [4, 10], "even" : True}, 
                    5:  {"name" : "pmm",  "layer_no" : 23, "no_atoms" : [4, 16], "even" : False},
                    6:  {"name" : "pmg",  "layer_no" : 24, "no_atoms" : [4, 16], "even" : True},
                    7:  {"name" : "pgg",  "layer_no" : 25, "no_atoms" : [4, 16], "even" : True},
                    8:  {"name" : "cm",   "layer_no" : 13, "no_atoms" : [4, 16], "even" : True},
                    9:  {"name" : "cmm",  "layer_no" : 26, "no_atoms" : [4, 16], "even" : True},
                    10: {"name" : "p4",   "layer_no" : 49, "no_atoms" : [4, 20], "even" : True},
                    11: {"name" : "p4m",  "layer_no" : 55, "no_atoms" : [4, 20], "even" : True},
                    12: {"name" : "p4g",  "layer_no" : 56, "no_atoms" : [4, 20], "even" : True},
                    13: {"name" : "p3",   "layer_no" : 65, "no_atoms" : [4, 20], "even" : True},
                    14: {"name" : "p3m1", "layer_no" : 69, "no_atoms" : [4, 20], "even" : True},
                    15: {"name" : "p31m", "layer_no" : 70, "no_atoms" : [4, 20], "even" : True},
                    16: {"name" : "p6",   "layer_no" : 73, "no_atoms" : [6, 20], "even" : True},
                    17: {"name" : "p6m",  "layer_no" : 77, "no_atoms" : [6, 20], "even" : True}}

sorted_groups = {"p1": 0, "p2": 1, "pg": 2, "pm": 3, "cm": 4, "pgg": 5, "pmg": 6, "pmm": 7, "cmm": 8,
                 "p4": 9, "p4g": 10, "p4m": 11, "p3": 12, "p3m1": 13, "p31m": 14, "p6": 15, "p6m": 16}


plane_symmetry_operations = {# point symmetry operations
                            0: np.array([[1,0,0],[0,1,0]]),       # {1|0}  
                            1: np.array([[-1,0,0],[0,-1,0]]),     # {2|0}
                            2: np.array([[-1,1,0],[-1,0,0]]),     # {3-|0} 
                            3: np.array([[0,-1,0],[1,-1,0]]),     # {3+|0}
                            4: np.array([[0,1,0],[-1,0,0]]),      # {4-|0}
                            5: np.array([[0,-1,0],[1,0,0]]),      # {4+|0} 
                            6: np.array([[0,1,0],[-1,1,0]]),      # {6-|0}
                            7: np.array([[1,-1,0],[1,0,0]]),      # {6+|0} 
                            8: np.array([[-1,0,0],[0,1,0]]),      # {m10|0}
                            9: np.array([[1,0,0],[0,-1,0]]),      # {m01,0}
                            10: np.array([[0,1,0],[1,0,0]]),      # {m1-1|0}
                            11: np.array([[0,-1,0],[-1,0,0]]),    # {m11,0}
                            12: np.array([[1,-1,0],[0,-1,0]]),    # {m12|0}
                            13: np.array([[-1,0,0],[-1,1,0]]),    # {m21|0}
                            # additional translations (=plane symmetry which are not point-like)
                            14: np.array([[1,0,0.5],[0,1,0.5]]),  # {1|.5,.5} 
                            15: np.array([[-1,0,0.5],[0,-1,0.5]]),# {2|.5,.5}
                            16: np.array([[-1,0,0.5],[0,1,0]]),   # {m10,.5,0}
                            17: np.array([[-1,0,0],[0,1,0.5]]),   # {m10|0,.5}
                            18: np.array([[-1,0,0.5],[0,1,0.5]]), # {m10|.5,.5}
                            19: np.array([[1,0,0.5],[0,-1,0]]),   # {m01|.5,0}
                            20: np.array([[1,0,0.5],[0,-1,0.5]]), # {m01|.5,.5}
                            21: np.array([[0,1,0.5],[1,0,0.5]]),  # {m1-1|.5,.5}
                            22: np.array([[0,-1,0.5],[-1,0,0.5]])}# {m11,.5,.5}


plane_symmetry_names = {0: "{1|0}", 
                        1: "{2|0}", 
                        2: "{3-|0}", 
                        3: "{3+|0}", 
                        4: "{4-|0}", 
                        5: "{4+|0}", 
                        6: "{6-|0}", 
                        7: "{6+|0}", 
                        8: "{m10|0}", 
                        9: "{m01,0}", 
                        10: "{m1-1|0}", 
                        11: "{m11,0}", 
                        12: "{m12|0}", 
                        13: "{m21|0}",
                        14: "{1|.5,.5}", 
                        15: "{2|.5,.5}", 
                        16: "{m10,.5,0}", 
                        17: "{m10|0,.5}", 
                        18: "{m10|.5,.5}", 
                        19: "{m01|.5,0}", 
                        20: "{m01|.5,.5}", 
                        21: "{m1-1|.5,.5}", 
                        22: "{m11|.5,.5}"}

wallpaper_symmetries = {"p1":  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p2":  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "pg":  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,1,0,0,0,0,0], 
                        "pm":  [1,0,0,0,0,0,0,0,1,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "cm":  [1,0,0,0,0,0,0,0,1,0,0,0,0,0,  1,0,0,0,1,0,0,0,0],
                        "pgg": [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,0,0],
                        "pmg": [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,1,0,0,1,0,0,0],
                        "pmm": [1,1,0,0,0,0,0,0,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "cmm": [1,1,0,0,0,0,0,0,1,1,0,0,0,0,  1,1,0,0,1,0,1,0,0],
                        "p4":  [1,1,0,0,1,1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p4g": [1,1,0,0,1,1,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,1,1],
                        "p4m": [1,1,0,0,1,1,0,0,1,1,1,1,0,0,  0,0,0,0,0,0,0,0,0],
                        "p3":  [1,0,1,1,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p3m1":[1,0,1,1,0,0,0,0,1,1,0,1,0,0,  0,0,0,0,0,0,0,0,0],
                        "p31m":[1,0,1,1,0,0,0,0,0,0,1,0,1,1,  0,0,0,0,0,0,0,0,0],
                        "p6":  [1,1,1,1,0,0,1,1,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p6m": [1,1,1,1,0,0,1,1,1,1,1,1,1,1,  0,0,0,0,0,0,0,0,0]}

plane_proper_symmetry = {0: "{1|0}",
                         1: "{2|0}", 
                         2: "{3|0}", 
                         3: "{4|0}",  
                         4: "{6|0}", 
                         5: "{m10|0}", 
                         6: "{m01,0}", 
                         7: "{m1-1|0}", 
                         8: "{m11,0}", 
                         9: "{m12|0}", 
                        10: "{m21|0}",
                        11: "{1|.5,.5}", 
                        12: "{2|.5,.5}", 
                        13: "{m10,.5,0}", 
                        14: "{m10|0,.5}", 
                        15: "{m10|.5,.5}", 
                        16: "{m01|.5,0}", 
                        17: "{m01|.5,.5}", 
                        18: "{m1-1|.5,.5}", 
                        19: "{m11|.5,.5}"}

symmetry_operations = {0: np.array([[1,0,0],[0,1,0]]),                              # {1|0}  
                       1: np.array([[-1,0,0],[0,-1,0]]),                            # {2|0}
                       2: np.array([[-1.,-np.sqrt(3.),0],[np.sqrt(3.),-1,0]])/2,    # {3|0} 
                       3: np.array([[0,-1,0],[1,0,0]]),                             # {4|0}
                       4: np.array([[ 1.,-np.sqrt(3.),0],[np.sqrt(3.), 1,0]])/2}    # {6|0}

wallpaper_reduced = {1: 1,
                    2:  1,
                    3:  2,
                    4:  2, 
                    5:  4,
                    6:  3,
                    7:  3,
                    8:  3,
                    9:  4,
                    10: 5,
                    11: 5,
                    12: 5,
                    13: 6,
                    14: 6,
                    15: 6,
                    16: 7,
                    17: 7}

gauss_k0 = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ]
                            
def random_wallpaper():
    return np.random.randint(17)+1


class RandomLattice:
    
    def __init__(self, supercell=[10,10,1], max_rotation_angle=90, atom_radius=.3, scale=1., path=None):
        self.path = path
        self.max_rotation_angle = max_rotation_angle
        self.supercell = supercell
        self.atom_radius = atom_radius
        self.scale = scale
        self.group = None
        
    def generate_lattice(self, wallpaper_class):
        
        self.group = wallpaper_groups[wallpaper_class]
        # randomize no of atoms in UC
        min_no_atoms_in_UC, max_no_atoms_in_UC = self.group["no_atoms"]
        if self.group["even"]:
            rand_bounds = [int(min_no_atoms_in_UC/2), int(max_no_atoms_in_UC/2)]
            rand_multiplier = 2
        else:
            rand_bounds = [min_no_atoms_in_UC, max_no_atoms_in_UC]
            rand_multiplier = 1
        no_atoms_in_UC = np.random.randint(rand_bounds[0], rand_bounds[1]+1)*rand_multiplier
        # define random crystal
        xtal = pyxtal()
        xtal.from_random(2, self.group["layer_no"], ['C'], [no_atoms_in_UC], thickness=2.)  # [no_atoms_in_UC]
        ase_lattice = xtal.to_ase()*self.supercell
        if self.max_rotation_angle > 1:
            ase_lattice.rotate(np.random.randint(self.max_rotation_angle)+1, 'z')
        return ase_lattice, self.group["name"]

    def render_lattice(self, ase_lattice, sample_name=None, title=False, return_ndarray=True):

        fig, ax = plt.subplots()
        fig.set_size_inches(5., 5.)
        plot_atoms(ase_lattice, ax, radii=self.atom_radius, scale=self.scale, show_unit_cell=2) 
        ax.set_axis_off()
        if title is False: fig.tight_layout()
        if return_ndarray:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=800)
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3])*8, int(fig.bbox.bounds[2])*8, -1))
            img_arr = img_arr[1000:3000,1000:3000]
            io_buf.close()
        if sample_name is not None:
            if title:
                ax.set_title(self.group["name"])
            fig.savefig(os.path.join(self.path, sample_name), dpi=400)
        plt.close()
        if return_ndarray:
            return img_arr[:,:,0]
    
    def fourier_transform(self, real_image, sample_name=None):
        transform = np.abs(fftshift(fft2(real_image)))
        fig, ax = plt.subplots()
        fig.set_size_inches(5., 5.)
        ax.imshow(transform[400:600,400:600], cmap='gray_r')
        ax.set_axis_off()
        fig.savefig(os.path.join(self.path, sample_name+'_f'), dpi=100)
        
    def fourier_transform_(self, real_image, window=True):
        if window:
            #w = filters.window('blackman', real_image.shape)
            w = filters.window(('gaussian', 120.), real_image.shape)  # 100.
        else:
            w = 1.
        fft = fftshift(fft2(real_image*w, norm='forward'))    
        return np.real(fft), np.imag(fft)
    
    def fourier_transform_s(self, real_image, rsx=0, rsy=0):
        w = filters.window(('gaussian', 120.), real_image.shape)  # 100.
        w = shift(
            w,
            shift=(-rsx, -rsy),
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
        fft = fftshift(fft2(real_image*w, norm='forward'))    
        return np.real(fft), np.imag(fft)
    
    def dump_labels(self, labels):
        with open(os.path.join(self.path, 'labels.json'), 'w') as fp:
            json.dump(labels, fp)
    

class Plotting:
    
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
     
    def plot_lattice(self, crystal_render, filename='lattice.png'): 
        crystal_render_plot = crystal_render.copy()
        crystal_render_plot[1000,:] = 0
        crystal_render_plot[:,1000] = 0
        fig, ax = plt.subplots()
        fig.set_size_inches(5., 5.)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        ax.set_aspect(1.)
        ax.imshow(crystal_render_plot[500:1500,500:1500], cmap='gray')
        ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        #ax.set_axis_off()
        fig.savefig(os.path.join(self.path, filename), dpi=200, bbox_inches='tight')
        plt.close()
     
    def plot_fouriers(self, transform, filename='fourier.png', select=None):
        amp = np.amax(np.abs(transform))/3.
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(11., 5.)
        ((ax1,ax2)) = axs
        ax1.set_aspect(1.)
        if select is not None:
            cb = ax1.imshow(transform.real[select[0]:select[1],select[0]:select[1]], cmap='bwr', vmin=-amp, vmax=amp)
        else:
            cb = ax1.imshow(transform.real, cmap='bwr', vmin=-amp, vmax=amp)
        ax1.tick_params(axis="both", labelbottom=False, labelleft=False)
        #ax1.set_axis_off()
        ax1.set_title('real')
        ax2.set_aspect(1.)
        if select is not None:
            cb = ax2.imshow(transform.imag[select[0]:select[1],select[0]:select[1]], cmap='bwr', vmin=-amp, vmax=amp)
        else:
            cb = ax2.imshow(transform.imag, cmap='bwr', vmin=-amp, vmax=amp)            
        ax2.tick_params(axis="both", labelbottom=False, labelleft=False)
        #ax2.set_axis_off()
        ax2.set_title('imag')
        fig.colorbar(cb, ax=axs.ravel().tolist())
        fig.savefig(os.path.join(self.path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_phaseF(self, phi, filename='phaseF.png', select=None):
        fig, ax = plt.subplots()
        fig.set_size_inches(6., 5.)
        ax.set_aspect(1.)
        if select is not None:
            cb = ax.imshow(phi[select[0]:select[1],select[0]:select[1]], cmap='PiYG')
        else:
            cb = ax.imshow(phi, cmap='PiYG')          
        ax.set_axis_off()
        cbar = fig.colorbar(cb, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(self.path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_phaseF_grid(
        self,
        phi_grid,
        filename="phaseF_grid.png",
        select=None,
        shared_colorbar=True,
        symmetric_scale=False,
        labels=None,
    ):
        """
        Rysuje dwuwymiarowy grid map.

        Parameters
        ----------
        phi_grid : torch.Tensor lub np.ndarray
            Tensor o kształcie:
                (n_rows, n_cols, H, W)

        filename : str
            Nazwa zapisywanego pliku.

        select : tuple[int, int] lub None
            Opcjonalny zakres wycinania map:
                select=(start, end)

        shared_colorbar : bool
            Jeśli True, wszystkie mapy używają wspólnej skali kolorów.

        symmetric_scale : bool
            Jeśli True, skala kolorów jest symetryczna względem zera.
            Jest to zwykle wskazane dla cmap='PiYG'.
        """

        if hasattr(phi_grid, "detach"):
            phi_grid = phi_grid.detach().cpu().numpy()

        phi_grid = np.asarray(phi_grid)

        if phi_grid.ndim != 4:
            raise ValueError(
                "Oczekiwany tensor o kształcie "
                f"(n_rows, n_cols, H, W), otrzymano {phi_grid.shape}"
            )

        n_rows, n_cols, _, _ = phi_grid.shape
        if select is not None:
            start, end = select
            phi_grid = phi_grid[:, :, start:end, start:end]

        if shared_colorbar:
            if symmetric_scale:
                limit = np.nanmax(np.abs(phi_grid))
                vmin = -limit
                vmax = limit
            else:
                vmin = np.nanmin(phi_grid)
                vmax = np.nanmax(phi_grid)
        else:
            vmin = None
            vmax = None

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        last_image = None

        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row, col]
                current_map = phi_grid[row, col]

                if shared_colorbar:
                    current_vmin = vmin
                    current_vmax = vmax
                elif symmetric_scale:
                    limit = np.nanmax(np.abs(current_map))
                    current_vmin = -limit
                    current_vmax = limit
                else:
                    current_vmin = np.nanmin(current_map)
                    current_vmax = np.nanmax(current_map)

                last_image = ax.imshow(
                    current_map,
                    cmap="PiYG",
                    vmin=current_vmin,
                    vmax=current_vmax,
                    origin="upper",
                )
                if labels is not None:
                    ax.set_title(f"({labels[row, col]:.3f})")
                else:
                    ax.set_title(f"({row}, {col})")
                ax.set_aspect("equal")
                ax.set_axis_off()

                if not shared_colorbar:
                    fig.colorbar(
                        last_image,
                        ax=ax,
                        shrink=0.8,
                    )

        if shared_colorbar and last_image is not None:
            fig.colorbar(
                last_image,
                ax=axes.ravel().tolist(),
                shrink=0.8,
                label="Value",
            )

        output_path = os.path.join(self.path, filename)

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)