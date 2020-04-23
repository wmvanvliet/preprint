"""
PyToch Dataset's for which the data is generated on the fly.
"""
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
from torchvision.datasets import VisionDataset

consonants = list('bcdfghjklmnpqrstvwxz')

rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(7, 16, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
noise_levels = [0.2, 0.35, 0.5]
lengths = [4, 5, 6]

fonts = {
    'ubuntu mono': [None, 'data/fonts/UbuntuMono-R.ttf'],
    'courier': [None, 'data/fonts/courier.ttf'],
    'luxi mono regular': [None, 'data/fonts/luximr.ttf'],
    'lucida console': [None, 'data/fonts/LucidaConsole-R.ttf'],
    'lekton': [None, 'data/fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': [None, 'data/fonts/DejaVuSansMono.ttf'],
    'times new roman': [None, 'data/fonts/times.ttf'],
    'arial': [None, 'data/fonts/arial.ttf'],
    'arial black': [None, 'data/fonts/arialbd.ttf'],
    'verdana': [None, 'data/fonts/verdana.ttf'],
    'comic sans ms': [None, 'data/fonts/comic.ttf'],
    'georgia': [None, 'data/fonts/georgia.ttf'],
    'liberation serif': [None, 'data/fonts/LiberationSerif-Regular.ttf'],
    'impact': [None, 'data/fonts/impact.ttf'],
    'roboto condensed': [None, 'data/fonts/Roboto-Light.ttf'],
}


class ConsonantStrings(VisionDataset):
    """A dataset that generates pictures of random consonant strings."""
    def __init__(self, length=100_000, width=64, height=64, label=0,
                 seed=0, labels='int', transform=None, target_transform=None):
        super().__init__('consonants', transform=transform,
                         target_transform=target_transform)
        self.length = length
        self.width = width
        self.height = height
        self.label = label
        self.classes = ['consonants']
        self.class_to_idx = {'consonants': 0}
        self.vectors = np.zeros((1, 300))
        self.labels = labels

        # Create figure of desired size in pixels (assume dpi of 96)
        self.fig = Figure(figsize=(width / 96., height / 96.), dpi=96.)
        # We need the canvas object to get the bitmap data at the end
        self.canvas = FigureCanvasAgg(self.fig)
        self.rng = np.random.RandomState(seed)

    def __getitem__(self, index):
        """Create a single consonant string image"""
        # Initialize an empty figure
        self.fig.clf()
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        # Select properties for this image
        rotation = self.rng.choice(rotations)
        fontsize = self.rng.choice(sizes)
        font = self.rng.choice(list(fonts.keys()))
        noise_level = self.rng.choice(noise_levels)
        length = self.rng.choice(lengths)

        # Generate the consonant string
        text = ''.join(self.rng.choice(consonants, length))

        # Start with a grey background image
        background = plt.Rectangle((0, 0), self.width, self.height,
                                   facecolor=(0.5, 0.5, 0.5, 1.0), zorder=0)
        ax.add_patch(background)

        # Add noise to the image. Note the usage of the alpha parameter to
        # tweak the amount of noise
        noise_image = np.random.randn(self.width, self.height)
        ax.imshow(noise_image, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level,
                  zorder=1)
        
        # Add the text to the image in the selected font
        fontfamily, fontfile = fonts[font]
        fontprop = fm.FontProperties(family=fontfamily, fname=fontfile, size=fontsize)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                rotation=rotation, fontproperties=fontprop,
                alpha=1 - noise_level, zorder=2)

        # Remove tick marks and the like
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        # Render the image and create a PIL.Image from the pixel data
        self.canvas.draw()
        buffer, (width, height) = self.canvas.print_to_buffer()
        image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        # Construct target
        if self.labels == 'int':
            target = self.label
        else:
            target = np.zeros(300)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.length
