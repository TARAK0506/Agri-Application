from app import app
import unittest
import warnings
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import patch, MagicMock

class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_no_file(self):
        response = self.app.post('/predict', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No file part', response.data)

    def test_predict(self):
        # Suppress TensorFlow warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
        
        # Create a dummy image
        image = Image.new('RGB', (256, 256), color='red')
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = self.app.post('/predict', data={'file': (BytesIO(img_byte_arr), 'test.png')})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'prediction', response.data)

    def test_recommend_crop_missing_fields(self):
        response = self.app.post('/recommend', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Missing required fields', response.data)

    def test_recommend_crop(self):
        # Suppress scikit-learn warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        
        data = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        response = self.app.post('/recommend', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'recommended_crop', response.data)
        
        # Additional assertions to verify the response content
        response_json = response.get_json()
        self.assertIn('recommended_crop', response_json)
        self.assertIsInstance(response_json['recommended_crop'], str)
        
if __name__ == '__main__':
    unittest.main()
    
    