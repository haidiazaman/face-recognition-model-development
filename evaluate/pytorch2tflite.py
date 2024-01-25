import torch
import onnx
import torchvision
from model import *
import onnx_tf


load_model = '/home/jovyan/face-recognition-model-development/train/result/v1.0.0/weights/best_model.pt'
num_classes = 902 

model = FaceNet(num_classes=num_classes,pretrained='vggface2')
if load_model:
    model.load_state_dict(torch.load(os.path.join('./result', load_model), map_location = device))
    _ = model.eval()
    
# strip the last classifier layer here to use the only the 512-embeddings layer
# code
    
# Set  input shape of the model
input_shape = (1, 3, 512, 512)

# Export  PyTorch model to ONNX format
torch.onnx.export(model, torch.randn(input_shape), 'inceptionNet.onnx', opset_version=11)

# Load  ONNX model
onnx_model = onnx.load('inceptionNet.onnx')

# Convert ONNX model to TensorFlow format
tf_model = onnx_tf.backend.prepare(onnx_model)
# Export  TensorFlow  model 
tf_model.export_graph("inceptionNet.tf")

# conver tf model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model("inceptionNet.tf")
tflite_model = converter.convert()
open('inceptionNet.tflite', 'wb').write(tflite_model)