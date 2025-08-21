import numpy as np
from jaxtyping import Float

Image = Float[np.ndarray, "channels height width"]
