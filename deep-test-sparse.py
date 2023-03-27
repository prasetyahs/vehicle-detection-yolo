from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
onnx_filepath = "../model/yolo-version-3.onnx"
batch_size = 16

# Generate random sample input
inputs = generate_random_inputs(onnx_filepath, batch_size)

# Compile and run
engine = compile_model(onnx_filepath, batch_size)
outputs = engine.run(inputs)