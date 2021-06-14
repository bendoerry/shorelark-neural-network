pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Network {
    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propagrate(inputs);
        }

        inputs
    }
}

impl Layer {
    fn propagrate(&self, inputs: Vec<f32>) -> Vec<f32> {
        todo!()
    }
}
