BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  ██████╗  ██████╗  ██████╗  ██████╗ ██╗   ██╗             ║
║  ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝ ╚██╗ ██╔╝             ║
║  ██║  ██║██║   ██║██║  ███╗██║  ███╗ ╚████╔╝              ║
║  ██║  ██║██║   ██║██║   ██║██║   ██║  ╚██╔╝               ║
║  ██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝   ██║                ║
║  ╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝    ╚═╝                ║
║                                                           ║
║       N E U R A L   G L I T C H   E N G I N E             ║
║                   v0.1 - meltingscales                    ║
╚═══════════════════════════════════════════════════════════╝
"""


class Config:
    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    N_MFCC = 13
    CHUNK_DURATION = 0.5
    IMAGE_SIZE = (512, 512)
    LATENT_DIM = 64
    DECODER_CHANNELS = [64, 32, 16, 8, 3]  # ConvTranspose2d channel progression
    USE_GPU = False

    COLOR_PALETTES = {
        'vaporwave': [(255, 113, 206), (103, 58, 183), (0, 191, 255)],
        'cyberpunk': [(255, 0, 110), (0, 255, 255), (255, 255, 0)],
        'grayscale': [(i, i, i) for i in range(0, 256, 85)],
        'demoscene': [(0, 255, 0), (255, 0, 255), (0, 255, 255)],
        # Drakonix fursona: deep navy body, dark navy scales, bright purple wings,
        # lilac hair, cyan/teal bg, light cyan, medium purple, pale lavender, off-white claws
        'drakonix': [
            (13,  27,  42),   # deep navy body
            (26,  26,  53),   # dark navy scales
            (147, 51,  234),  # bright purple wings
            (192, 132, 252),  # light lilac hair
            (6,   182, 212),  # cyan/teal bg
            (34,  211, 238),  # light cyan
            (168, 85,  247),  # medium purple
            (216, 180, 254),  # pale lavender
            (240, 240, 240),  # off-white claws
        ],
    }

    EFFECT_WEIGHTS = {
        'neural': 0.3,
        'bitcrush': 0.2,
        'datamosh': 0.2,
        'rgb_split': 0.2,
        'chaos': 0.1,
    }
