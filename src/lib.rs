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
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers.iter().fold(inputs, | inputs, layer| layer.propagrate(inputs))
    }
}

impl Layer {
    fn propagrate(&self, inputs: Vec<f32>) -> Vec<f32> {
        todo!()
    }
}
