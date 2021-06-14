use crate::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    pub fn random(
        rng: &mut dyn rand::RngCore,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }
}

#[cfg(test)]
mod tests {
    mod random {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use super::super::Layer;

        #[test]
        fn test_biases() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            let actual_biases: Vec<_> = layer.neurons.iter().map(|neuron| neuron.bias).collect();
            let expected_biases = vec![-0.6255188, 0.5238807];

            approx::assert_relative_eq!(actual_biases.as_slice(), expected_biases.as_slice());
        }

        #[test]
        fn test_weights() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            let actual_weights: Vec<_> = layer
                .neurons
                .iter()
                .map(|neuron| neuron.weights.as_slice())
                .collect();
            let expected_weights: Vec<&[f32]> = vec![
                &[0.67383957, 0.8181262, 0.26284897],
                &[-0.53516835, 0.069369674, -0.7648182],
            ];

            approx::assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
        }
    }
}
