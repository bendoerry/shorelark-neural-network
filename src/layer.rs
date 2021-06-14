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
