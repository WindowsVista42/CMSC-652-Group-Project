import numpy as np
import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption, load_pem_private_key
from cryptography.hazmat.backends import default_backend
from PIL import Image
from io import BytesIO
from typing import List, Tuple

class MedicalImageWatermarker:
    def __init__(self, rsa_key_size=2048):
        self.rsa_key_size = rsa_key_size
        self.backend = default_backend()

    def generate_keys(self):
        """Generate RSA key pair using cryptography module"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.rsa_key_size,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        return (
            private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ),
            public_key.public_bytes(
                Encoding.PEM,
                PublicFormat.SubjectPublicKeyInfo
            )
        )

    # def _form_quads(self, image_data: np.ndarray) -> List[Tuple[int, int, int, int]]:
    #     """Forms non-overlapping quads from the image data."""
    #     height, width = image_data.shape
    #     quads = []
    #     for i in range(0, height - 1, 2):
    #         for j in range(0, width - 1, 2):
    #             quad = (
    #                 image_data[i, j],
    #                 image_data[i, j + 1],
    #                 image_data[i + 1, j],
    #                 image_data[i + 1, j + 1]
    #             )
    #             quads.append(quad)
    #     return quads

    def _is_in_roe(self, quad_index: int, image_width: int, roe_vertices: List[Tuple[int, int]]) -> bool:
        """Determine if a quad is within the Region of Embedding (ROE)"""
        # Calculate quad grid dimensions
        quad_grid_width = (image_width - 1) // 2
        if quad_grid_width <= 0:
            return False
        
        # Get quad position in grid
        row = quad_index // quad_grid_width
        col = quad_index % quad_grid_width
        
        # Calculate center coordinates (x, y) in image space
        x = col * 2 + 1  # Column (width dimension)
        y = row * 2 + 1  # Row (height dimension)
        
        return self._point_in_polygon((x, y), roe_vertices)

    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm to determine if a point is inside a polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1) % n]
            x1, y1 = p1
            x2, y2 = p2

            # Check if point is exactly on a vertical edge
            if x1 == x2 == x and (y >= min(y1, y2) and y <= max(y1, y2)):
                return True
            
            # Check if point is exactly on a horizontal edge
            if y1 == y2 == y and (x >= min(x1, x2) and x <= max(x1, x2)):
                return True
            
            # Check intersection with edge
            if ((y1 > y) != (y2 > y)):
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x <= xinters:
                    inside = not inside
                    
        return inside

    def _form_quads(self, image_data: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Create 2x2 pixel quads from image data"""
        height, width = image_data.shape
        quads = []
        
        for row in range(0, height - 1, 2):
            for col in range(0, width - 1, 2):
                quad = (
                    image_data[row, col],
                    image_data[row, col + 1],
                    image_data[row + 1, col],
                    image_data[row + 1, col + 1]
                )
                quads.append(quad)
        
        return quads

    def _reconstruct_image(self, original_data: np.ndarray, 
                      embedded_quads: List[Tuple[int, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Reconstructs watermarked image from modified quads while preserving medical data integrity"""
        # Create copy of original image data
        reconstructed = np.copy(original_data)
        height, width = original_data.shape
        
        # Calculate quad grid dimensions
        quad_grid_width = (width // 2) - 1
        if quad_grid_width <= 0:
            return original_data
        
        for quad_index, modified_quad in embedded_quads:
            # Calculate quad position in image space
            row = (quad_index // quad_grid_width) * 2
            col = (quad_index % quad_grid_width) * 2
            
            # Ensure we stay within image boundaries
            if row + 1 >= height or col + 1 >= width:
                continue

            #Get the inverse difference expansion and embed that, not the modified quad itself

            inverse_modified_quad = self._inverse_difference_expansion(modified_quad)
            
            # Update pixel values with modified quad
            # reconstructed[row, col] = modified_quad[0]
            # reconstructed[row, col+1] = modified_quad[1]
            # reconstructed[row+1, col] = modified_quad[2]
            # reconstructed[row+1, col+1] = modified_quad[3]

            reconstructed[row, col] = inverse_modified_quad[0]
            reconstructed[row, col+1] = inverse_modified_quad[1]
            reconstructed[row+1, col] = inverse_modified_quad[2]
            reconstructed[row+1, col+1] = inverse_modified_quad[3]
        
        return reconstructed


    def _compute_hash(self, image_data: bytes) -> str:
        """Compute MD5 hash of image data as specified in the paper"""
        return hashlib.md5(image_data).hexdigest()

    def _sign_data(self, data: bytes, private_key: bytes) -> bytes:
        """Create digital signature using RSA-PSS"""
        key = load_pem_private_key(private_key, None, self.backend)
        # print(data, type(data))
        return key.sign(
            data.encode("UTF-8"),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

    def _verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify digital signature using RSA-PSS"""
        key = rsa.load_pem_public_key(public_key, self.backend)
        try:
            key.verify(
                signature,
                data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def _difference_expansion(self, quad: Tuple[int, int, int, int]):
        """Apply difference expansion transform"""
        # u0, u1, u2, u3 = quad
        # quad16 = quad.astype(uint16)
        u0, u1, u2, u3 = quad

        u0 = u0.astype(np.int64)
        u1 = u1.astype(np.int64)
        u2 = u2.astype(np.int64)
        u3 = u3.astype(np.int64)

        v0 = (u0 + u1 + u2 + u3) // 4
        # v0 = (u0.astype(np.int64) + u1.astype(np.int64) + u2.astype(np.int64) + u3.astype(np.int64)) // 4
        # print("Quad: ", quad, "v0: ", v0, type(quad), type(u0), type(v0))
        return (v0, (u1 - u0), (u2 - u0), (u3 - u0))

    def _inverse_difference_expansion(self, transformed_quad: Tuple[int, int, int, int]):
        """Inverse difference expansion transform"""
        v0, dv1, dv2, dv3 = transformed_quad
        u0 = v0 - (dv1 + dv2 + dv3) // 4
        return (u0, u0 + dv1, u0 + dv2, u0 + dv3)

    def _embed_bits(self, transformed_quad: Tuple[int, int, int, int], bits: str):
        """Embed 3 bits into transformed quad"""
        # print(bits)
        v0, v1, v2, v3 = transformed_quad
        return (
            v0,
            (v1 << 1) | int(bits[0]),
            (v2 << 1) | int(bits[1]),
            (v3 << 1) | int(bits[2])
        )

    def _get_expandable_quads(self, quads: List[Tuple[int, int, int, int]]):

        # TODO: Code to obtain expandable quads as per Eq3 from paper

        expandable_quads = list()

        for quad in quads:

            u0, u1, u2, u3 = quad

            # if (u0 < 126) and (u1 < 126) and (u2 < 126) and (u3 < 126):
            if (u0 < 127) and (u1 < 127) and (u2 < 127) and (u3 < 127):
                expandable_quads.append(quad)

        return expandable_quads

    def _extract_bits(self, transformed_quad: Tuple[int, int, int, int]):
        """Extract 3 bits from transformed quad"""
        _, v1, v2, v3 = transformed_quad
        return f"{v1&1}{v2&1}{v3&1}"

    def embed_watermark(self, image_path: str, private_key: bytes, payload: str, roe_vertices: List[Tuple[int, int]]):
        """Main embedding workflow"""
        # Load and validate image
        img = Image.open(image_path).convert("L")
        # img = Image.open(image_path)
        print("Image Mode: ", img.mode)
        # img_data = np.array(img, dtype=np.int16)
        img_data = np.array(img)
        print("Image Shape: ", img_data.shape, type(img_data))
        
        # Cryptographic operations
        img_bytes = img.tobytes()
        content_hash = self._compute_hash(img_bytes)
        # signature = self._sign_content(content_hash, private_key)
        signature = self._sign_data(content_hash, private_key)
        
        # Prepare payload (content hash + signature + EPR data)
        full_payload = f"{content_hash}||{signature.hex()}||{payload}"
        binary_payload = ''.join(f"{ord(c):08b}" for c in full_payload)

        # print("Full Payload: ", full_payload, "Binary Payload: ", binary_payload)
        
        # Implement ROE-based embedding using difference expansion
        quads = self._form_quads(img_data)

        #TODO Expandable quads
        expandable_quads = self._get_expandable_quads(quads)

        embedded_quads = []
        print(len(quads), len(expandable_quads))
        
        for idx, quad in enumerate(quads):
            if self._is_in_roe(idx, img.width, roe_vertices):
                transformed = self._difference_expansion(quad)
                if len(binary_payload) >= 3:
                    bits = binary_payload[:3]
                    embedded = self._embed_bits(transformed, bits)
                    binary_payload = binary_payload[3:]
                    embedded_quads.append((idx, embedded))
        
        # Reconstruct watermarked image
        print(len(embedded_quads))
        watermarked = self._reconstruct_image(img_data, embedded_quads)
        return Image.fromarray(watermarked)

    def extract_watermark(self, image_path: str, public_key: bytes, roe_vertices: List[Tuple[int, int]]):
        """Main extraction workflow"""
        img = Image.open(image_path).convert("L")
        img_data = np.array(img)
        
        # Extract payload
        quads = self._form_quads(img_data)
        extracted_bits = []
        
        for idx, quad in enumerate(quads):
            if self._is_in_roe(idx, img.width, roe_vertices):
                transformed = self._difference_expansion(quad)
                extracted_bits.append(self._extract_bits(transformed))
        
        # Reconstruct and validate payload
        full_payload = ''.join(extracted_bits)
        # ... (payload parsing and validation logic)
        
        return full_payload
