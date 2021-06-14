use crate::layer::Layer;
use crate::layer_topology::LayerTopology;

mod layer;
mod layer_topology;
mod neuron;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn random(rng: &mut dyn rand::RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }
}
