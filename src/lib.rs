#![feature(crate_visibility_modifier)]
#![feature(array_methods)]

use std::iter::once;

use crate::layer::Layer;
pub use crate::layer_topology::LayerTopology;

mod layer;
mod layer_topology;
mod neuron;

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    crate fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

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

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .cloned()
    }

    pub fn from_weights(layers: &[LayerTopology], weights: impl IntoIterator<Item = f32>) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| Layer::from_weights(layers[0].neurons, layers[1].neurons, &mut weights))
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }
}

#[cfg(test)]
mod tests {

    mod random {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use crate::layer_topology::LayerTopology;

        use super::super::Network;

        #[test]
        fn test_layer_sizes() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 3 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 1 },
                ],
            );

            assert_eq!(network.layers.len(), 2);
            assert_eq!(network.layers[0].neurons.len(), 2);
            assert_eq!(network.layers[1].neurons.len(), 1);
        }

        #[test]
        fn test_biases() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 3 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 1 },
                ],
            );

            approx::assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
            approx::assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);
            approx::assert_relative_eq!(network.layers[1].neurons[0].bias, -0.102499366);
        }

        #[test]
        fn test_weights() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 3 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 1 },
                ],
            );

            approx::assert_relative_eq!(
                network.layers[0].neurons[0].weights.as_slice(),
                &[0.67383957, 0.8181262, 0.26284897].as_slice()
            );
            approx::assert_relative_eq!(
                network.layers[0].neurons[1].weights.as_slice(),
                &[-0.5351684, 0.069369555, -0.7648182].as_slice()
            );
            approx::assert_relative_eq!(
                network.layers[1].neurons[0].weights.as_slice(),
                &[-0.48879623, -0.19277143].as_slice()
            );
        }
    }

    mod propagate {
        use crate::{layer::Layer, neuron::Neuron};

        use super::super::Network;

        #[test]
        fn test() {
            let layers = (
                Layer::new(vec![
                    Neuron::new(0.0, vec![-0.5, -0.4, -0.3]),
                    Neuron::new(0.0, vec![-0.2, -0.1, 0.0]),
                ]),
                Layer::new(vec![Neuron::new(0.0, vec![-0.5, 0.5])]),
            );
            let network = Network::new(vec![layers.0.clone(), layers.1.clone()]);

            let actual = network.propagate(vec![0.5, 0.6, 0.7]);
            let expected = layers.1.propagate(layers.0.propagate(vec![0.5, 0.6, 0.7]));

            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
        }
    }

    mod weights {
        use crate::{layer::Layer, neuron::Neuron};

        use super::super::Network;

        #[test]
        fn test() {
            let network = Network::new(vec![
                Layer::new(vec![Neuron::new(0.1, vec![0.2, 0.3, 0.4])]),
                Layer::new(vec![Neuron::new(0.5, vec![0.6, 0.7, 0.8])]),
            ]);

            let actual: Vec<_> = network.weights().collect();
            let expected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
        }
    }

    mod from_weights {
        use crate::LayerTopology;

        use super::super::Network;

        #[test]
        fn test() {
            let layers = &[LayerTopology { neurons: 3 }, LayerTopology { neurons: 2 }];

            let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            let network = Network::from_weights(layers, weights.clone());
            let actual: Vec<_> = network.weights().collect();

            approx::assert_relative_eq!(actual.as_slice(), weights.as_slice());
        }
    }
}
