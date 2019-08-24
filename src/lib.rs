//! Implementation of multilayer perceptron,backpropagation algorithm and some kind of genetic algorithm.
//! Using genetic algorithm:
//! ```rust
//! extern crate maver;
//! use maver::*;
//!
//!struct XOR;
//!
//!impl Environment for XOR {
//!    fn test(&self, org: &mut Organism) {
//!        let mut distance: f64;
//!        let nn = &mut org.nn;
//!        let out = nn.forward_prop(&[0.0, 0.0]);
//!        distance = (0.0 - out[0]).powi(2);
//!        let out = nn.forward_prop(&[0.0, 1.0]);
//!        distance += (1.0 - out[0]).powi(2);
//!        let out = nn.forward_prop(&[1.0, 0.0]);
//!        distance += (1.0 - out[0]).powi(2);
//!        let out = nn.forward_prop(&[1.0, 1.0]);
//!        distance += (0.0 - out[0]).powi(2);
//!        let fitness = 16.0 / (1.0 + distance);
//!        org.fitness = fitness;
//!    }
//!}
//!
//!fn main() {
//!    let mut n = Genetic::new(100, &[2, 1], Activation::Tanh, Activation::Tanh);
//!    let p = LearnParams::default();
//!    let mut champion = None;
//!    let mut i = 0;
//!    while champion.is_none() {
//!        n.evaluate(&mut XOR);
//!        if n.get_champion().fitness > 15.9 {
//!            champion = Some(n.get_champion().clone());
//!        }
//!        println!(
//!            "Iteration: {:?},best fitness: {}",
//!           i,
//!            n.get_champion().fitness
//!        );
//!        n.evolve(&p);
//!
//!       i += 1;
//!    }
//!    println!("{:#?}", champion.as_ref().unwrap());
//!    println!(
//!        "{:?}",
//!        champion.as_mut().unwrap().nn.forward_prop(&[0.0, 0.0])
//!    );
//!    println!(
//!        "{:?}",
//!        champion.as_mut().unwrap().nn.forward_prop(&[1.0, 0.0])
//!    );
//!    println!(
//!        "{:?}",
//!        champion.as_mut().unwrap().nn.forward_prop(&[1.0, 1.0])
//!    );
//!    println!(
//!        "{:?}",
//!        champion.as_mut().unwrap().nn.forward_prop(&[0.0, 1.0])
//!    );
//!}
//! ```
//! 
//! Using backpropagation algorithm:
//! ```rust
//! let mut nn = NeuralNetwork!(vec![2,2,1],Activation::Tanh,Activation::Tanh);
//! 
//!let xor_sets = vec![
//!    (vec![0., 0.], vec![0.0]),
//!    (vec![0., 1.], vec![1.0]),
//!    (vec![1., 0.], vec![1.0]),
//!    (vec![1., 1.], vec![0.0]),
//!];
//! 
//! nn.train(1000,false,xor_sets,Some(0.01));
//! println!("{:?}",nn.forward_prop(&[0.0,0.0]));
//! println!("{:?}",nn.forward_prop(&[0.0,1.0]));
//! ```

pub trait RegularizationFn {
    fn output(&self, _: f64) -> f64;
    fn der(&self, _: f64) -> f64;
}

use serde::{Deserialize,Serialize};

#[derive(Copy, Clone)]
pub struct L1;

impl RegularizationFn for L1 {
    #[inline(always)]
    fn output(&self, w: f64) -> f64 {
        w.abs()
    }
    fn der(&self, w: f64) -> f64 {
        if w < 0.0 {
            -1.0
        } else {
            if w > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct L2;
impl RegularizationFn for L2 {
    fn output(&self, w: f64) -> f64 {
        0.5 * w * w
    }
    #[inline(always)]
    fn der(&self, w: f64) -> f64 {
        w
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Regularization {
    L1,
    L2,
}

impl Regularization {
    pub fn output(self, x: f64) -> f64 {
        match self {
            Regularization::L1 => L1::output(&L1, x),
            Regularization::L2 => L2::output(&L2, x),
        }
    }
    pub fn der(self, x: f64) -> f64 {
        match self {
            Regularization::L1 => L1::der(&L1, x),
            Regularization::L2 => L2::der(&L2, x),
        }
    }
}



macro_rules! activation_fn {
    (
        $(
        $name: ident {
            output $x: ident => $e: block,
            der $x2: ident => $e2: block
        }
        ),*
    ) => {
        #[derive(Copy,Clone,PartialEq,Eq,Debug,Serialize,Deserialize)]
        pub enum Activation {
            $(
                $name
            ),*
        }

        impl Activation {
            pub fn output(self,x: f64) -> f64 {
                match self {
                    $(Activation::$name =>{let $x = x; $e}),*
                }
            }
            pub fn der(self,x: f64) -> f64 {
                match self {
                    $(Activation::$name => {let $x2 = x;$e2}),*
                }
            }
        }

    };
}

activation_fn! {
    Tanh {
        output x => {x.tanh().abs()},
        der x => {
            let output = x.tanh().abs();
            return 1.0 - output * output;
        }
    },
    Arctan {
        output x => {
            x.atan()
        },
        der x => {
            (1.0 / (x.powi(2) + 1.0))
        }
    },
    SoftSign {
        output x => {
            let x = x / (1.0 + x.abs());
            x
        },
        der x => {
            let x = x / (1.0 + x.abs()).powi(2);
            x
        }
    },
    SoftPlus {
        output x => {
            let x = 1.0 + std::f64::consts::E.powf(x);
            x.ln()
        },
        der x => {
            let x = 1.0 / (1.0 + std::f64::consts::E.powf(x));
            x
        }
    },
    BentIdentity {
        output x => {
            let x = ((x.powi(2) + 1.0).sqrt() - 1.0 / 2.0) + x;
            x
        },  
        der x => {
            let x = (x / (2.0*(x.powi(2) + 1.0).sqrt())) + 1.0;
            x
        }
    },
    Gaussian {
        output x => {
            let x = std::f64::consts::E.powf(-x.powi(2));
            x
        },
        der x => {
            let x = -2.0 * x * std::f64::consts::E.powf(-x.powi(2));
            x
        }
    },
    Relu {
        output x => {
            if 0.0 > x {
                0.0
            } else {
                x
            }
        },
        der x => {
            if x <= 0.0 {
                0.0
            } else {
                1.0
            }
        }
    },
    Sigmoid {
        output x => {
            1.0 / (1.0 + (-x).exp())
        },
        der x => {
            let output = 1.0 / (1.0 + (-x).exp());
            return output * (1.0-output)
        }
    },
    Linear {
        output x => {x},
        der _x => {1.0}
    }
}

pub fn err_fn(x: f64, y: f64) -> f64 {
    0.5 * (x - y).powi(2)
}
pub fn err_der(x: f64, y: f64) -> f64 {
    x - y
}

use rand::random;

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct Perceptron {
    pub delta: f64,
    pub output: f64,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: Activation,
}
impl Perceptron {
    fn create(inputs: usize, activation: Activation) -> Perceptron {
        Perceptron {
            weights: (0..inputs).map(|_|{let mut rng = rand::thread_rng(); rng.gen_range(-1.0f64,1.0f64)}).collect(),
            output: 0.0,
            delta: 0.0,
            bias: random::<f64>(),
            activation,
        }
    }
    /// Activate perceptron and get output
    pub fn activate(&self, inputs: &Vec<f64>) -> f64 {
        self.activation.output(if self.weights.len() < 30 {
            self.weights
                .iter()
                .zip(inputs.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f64>()
                + self.bias
            } else {
                use rayon::iter::*;
                self.weights
                    .par_iter()
                    .zip(inputs.par_iter())
                    .map(|(weight,input)| weight * input)
                    .sum::<f64>() + self.bias
            }
        )
    }

    pub fn derivative(&self) -> f64 {
        self.activation.der(self.output)
        //self.output * (1.0 - self.output)
    }
}

type Layer = Vec<Perceptron>;

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    layer: Vec<usize>,
}
impl NeuralNetwork {
    /// Create new neural network
    /// `layer_sizes`: neural network layout.
    /// `activation`: activation function for hidden perceptrons
    /// `output_act`: activation function for output perceptrons
    pub fn new(
        layer_sizes: Vec<usize>,
        activation: Activation,
        output_act: Activation,
    ) -> NeuralNetwork {
        NeuralNetwork {
            layers: (0..)
                .zip(layer_sizes[1..].iter())
                .map(|(i, layer)| {
                    (0..*layer)
                        .map(|_| {
                            let act = if i == layer_sizes.len() - 1 {
                                output_act
                            } else {
                                if activation == Activation::Linear {
                                    //panic!("Can't use linear activation in input/hidden layers");
                                }
                                activation
                            };
                            Perceptron::create(layer_sizes[i], act)
                        })
                        .collect()
                })
                .collect(),
            layer: layer_sizes,
        }
    }
    /// Get neural network output
    pub fn forward_prop(&self, row: &[f64]) -> Vec<f64> {
        use rayon::iter::*;
        self.layers.iter().fold(row.to_vec(), |inputs, layer| {
            layer
                .par_iter()
                .map(|perceptron| perceptron.activate(&inputs))
                .collect()
        })
    }
    /// Set & get neural network output (used for backpropagation)
    pub fn forward_set(&mut self, row: &Vec<f64>) -> Vec<f64> {
        use rayon::iter::*;
        self.layers.iter_mut().fold(row.clone(), |inputs, layer| {
            layer
                .par_iter_mut()
                .map(|perceptron| {
                    perceptron.output = perceptron.activate(&inputs);
                    perceptron.output
                })
                .collect()
        })
    }
    /// Train neural network
    /// `epochs`: how many times we will iterate
    /// `shuffle`: shuffle train data
    /// `data`: vector of training data: Vec<(input,expected output)>
    /// `lrate`: learning rate if none = 0.1
    pub fn train(
        &mut self,
        epochs: usize,
        shuffle: bool,
        mut data: Vec<(Vec<f64>, Vec<f64>)>,
        lrate: Option<f64>,
    ) {
        for _ in 0..epochs {
            for (input, expected) in data.iter() {
                self.forward_set(&input);
                self.backpropagate(&input, &expected, lrate);
            }
            if shuffle {
                let mut rng = rand::thread_rng();
                use rand::seq::SliceRandom;
                data.shuffle(&mut rng);
            }
        }
    }
    /// Backpropagation algorithm implementation
    /// `input`: input data
    /// `expected`: expected output
    /// `lrate`: learning rate if none = 0.1
    pub fn backpropagate(&mut self, input: &Vec<f64>, expected: &Vec<f64>, lrate: Option<f64>) {
        for i in (0..self.layers.len()).rev() {
            let prev: Vec<f64> = if i == 0 {
                input.clone()
            } else {
                self.layers[i - 1].iter().map(|x| x.output).collect()
            };
            for j in 0..self.layers[i].len() {
                let err = if i == self.layers.len() - 1 {
                    expected[j] - self.layers[i][j].output
                } else {
                    self.layers[i + 1]
                        .iter()
                        .map(|perceptron| perceptron.weights[j] * perceptron.delta)
                        .sum()
                };
                let ref mut perceptron = self.layers[i][j];
                perceptron.delta = err * perceptron.derivative();
                for k in 0..prev.len() {
                    perceptron.weights[k] += perceptron.delta * prev[k] * lrate.unwrap_or(0.1);
                }
                perceptron.bias += perceptron.delta * lrate.unwrap_or(0.1);
            }
        }
    }
    /// Create child from two neural networks
    fn reproduce(&self, other: &NeuralNetwork, fittest: bool) -> NeuralNetwork {
        let (best, worst) = if fittest {
            (self, other)
        } else {
            (other, self)
        };
        let mut nn = self.clone();
        for (i, layer) in nn.layers.iter_mut().enumerate() {
            if rand::random::<f64>() < 0.5 {
                *layer = best.layers[i].clone();
            } else {
                *layer = worst.layers[i].clone();
            }
        }
        nn
    }

    pub fn save_to_file(&self,file_path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self).unwrap();
        std::fs::write(file_path,&json)
    } 
}

#[derive(Clone, Debug)]
pub struct Organism {
    pub nn: NeuralNetwork,
    pub fitness: f64,
}

trait VecUtil<T> {
    fn remove_random(&mut self);
    fn insert_at_random(&mut self, _: T);
    fn insert_at_random_exclude(&mut self, _: T, _: &[usize]) {}
    fn remove_random_exclude(&mut self, _: &[usize]) {}
}
use rand::Rng;

impl<T> VecUtil<T> for Vec<T> {
    fn remove_random(&mut self) {
        let mut rng = rand::thread_rng();
        if self.len() == 1 {
            self.remove(0);
            return;
        }
        let idx = rng.gen_range(0, self.len());
        self.remove(idx);
    }
    fn insert_at_random(&mut self, item: T) {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0, self.len());
        self.insert(idx, item);
    }
}

impl Organism {
    /// Mutate organism
    /// This function may do next things:
    /// - Add new perceptron to random layer.
    /// - Remove perceptron from random layer.
    /// - Mutate bias.
    /// - Mutate weights.
    pub fn mutate(&mut self, learn_params: &LearnParams) {
        use rand::distributions::Distribution;
        use rand_distr::Normal;
        if random::<f64>() < learn_params.mutate_add_neuron_pr {
            if self.nn.layers.len() == 2 {
                /*let icount = self.nn.layers[0][0].weights.len();
                let act = self.nn.layers[0][0].activation;
                let p = Perceptron::create(icount, act);
                self.nn.layers.insert(1, vec![p]);*/
            } else {
                let sacred_layers = self.nn.layer[0] + self.nn.layer[self.nn.layer.len() - 1];
                if !(self.nn.layers.len() <= sacred_layers) {
                    let lidx =
                        random::<usize>() % (self.nn.layers.len() - sacred_layers) + sacred_layers;
                    let pidx = random::<usize>() % self.nn.layers[lidx].len();
                    let icount = self.nn.layers[lidx - 1][0].weights.len();
                    let act = self.nn.layers[lidx - 1][0].activation;
                    let p = Perceptron::create(icount, act);
                    self.nn.layers[lidx].insert(pidx, p);
                }
            }
        }
        if random::<f64>() < learn_params.mutate_del_neuron_pr {
            let sacred_layers = self.nn.layer[0] + self.nn.layer[self.nn.layer.len() - 1];
            if !(self.nn.layers.len() <= sacred_layers) {
                let lidx =
                    random::<usize>() % (self.nn.layers.len() - sacred_layers) + sacred_layers;
                let pidx = random::<usize>() % self.nn.layers[lidx].len();
                self.nn.layers[lidx].remove(pidx);
                if self.nn.layers[lidx].is_empty() {
                    self.nn.layers[lidx].remove(0);
                }
            }
        }
        let bias_distr = Normal::new(0.0, learn_params.bias_mutate_var).unwrap();
        let weight_distr = Normal::new(0.0, learn_params.weight_mutate_var).unwrap();

        let mut rng = rand::thread_rng();
        self.nn.layers.iter_mut().for_each(|x: &mut Layer| {
            x.iter_mut().for_each(|x: &mut Perceptron| {
                if random::<f64>() < learn_params.bias_mutate_pr {
                    x.bias += bias_distr.sample(&mut rng);
                } else if random::<f64>() < learn_params.bias_replace {
                    if random::<f64>() < learn_params.bias_zero_pr {
                        x.bias = 0.0;
                    } else if random::<f64>() < learn_params.bias_one_pr {
                        x.bias = 1.0;
                    } else {
                        x.bias = bias_distr.sample(&mut rng);
                    }
                } /*else if random::<f64>() < BIAS_ZERO_PR {
                      x.bias = 0.0;
                  } else if random::<f64>() < BIAS_ONE_PR {
                      x.bias = 1.0;
                  }*/
                x.weights.iter_mut().for_each(|weight: &mut f64| {
                    if random::<f64>() < learn_params.weight_mutate_pr {
                        *weight += weight_distr.sample(&mut rng);
                    } else if random::<f64>() < learn_params.weight_replace_pr {
                        *weight += weight_distr.sample(&mut rng);
                    }
                });
            })
        });
    }
    /// Get child from two organisms    
    pub fn mate(&self, fittest: bool, other: &Organism, p: &LearnParams) -> Organism {
        let mut org = Organism {
            fitness: 0.0,
            nn: self.nn.reproduce(&other.nn, fittest),
        };
        if random::<f64>() < 0.5 {
            org.mutate(&p);
        }
        org
    }
}
pub struct Genetic {
    organisms: Vec<Organism>,
}

/// A trait that is implemented by user to allow test of the Environment.
pub trait Environment: Send + Sync {
    /// This test will set the organism fitness:
    /// ```rust
    /// impl Environment for MyEnv {
    ///     fn test(&self,organism: &mut Organism) {
    ///         ...
    ///         organism.fitness = fitness;
    ///     }
    /// }
    ///
    /// ```
    fn test(&self, _: &mut Organism);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct LearnParams {
    pub mutate_add_neuron_pr: f64,
    pub weight_mutate_var: f64,
    pub weight_mutate_pr: f64,
    pub weight_replace_pr: f64,
    pub bias_zero_pr: f64,
    pub bias_one_pr: f64,
    pub bias_mutate_var: f64,
    pub bias_mutate_pr: f64,
    pub bias_replace: f64,
    pub mutate_del_neuron_pr: f64,
}

impl Default for LearnParams {
    fn default() -> Self {
        LearnParams {
            mutate_add_neuron_pr: MUTATE_ADD_NEURON_PR,
            weight_mutate_var: WEIGHT_MUTATE_VAR,
            weight_mutate_pr: WEIGHT_MUTATE_PR,
            weight_replace_pr: WEIGHT_REPLACE_PR,
            bias_zero_pr: BIAS_ZERO_PR,
            bias_one_pr: BIAS_ONE_PR,
            bias_mutate_pr: BIAS_MUTATE_PR,
            bias_mutate_var: BIAS_MUTATE_VAR,
            bias_replace: BIAS_REPLACE_PR,
            mutate_del_neuron_pr: MUTATE_DEL_NEURON_PR,
        }
    }
}

const MUTATE_ADD_NEURON_PR: f64 = 0.018564851821478344;
const WEIGHT_MUTATE_VAR: f64 = 0.8539035934199557;
const WEIGHT_MUTATE_PR: f64 = 0.2938496323249987;
const WEIGHT_REPLACE_PR: f64 = 0.020513723672854978;
const BIAS_ZERO_PR: f64 = 0.1568246658042563;
const BIAS_ONE_PR: f64 = 0.1868246658042563;
const BIAS_MUTATE_VAR: f64 = 0.25153760530420227;
const BIAS_MUTATE_PR: f64 = 0.2568246658042563;
const BIAS_REPLACE_PR: f64 = 0.13720985010407194;
const MUTATE_DEL_NEURON_PR: f64 = 0.018564851821478344;

impl Genetic {
    /// Create new genetic algorithm implementation
    /// `population_size`: count of organisms in one generation
    /// `network_init`: network topology
    /// `act_in:`: activation function for hidden perceptrons
    /// `act_out`: activation function for output perceptrons
    pub fn new(
        population_size: usize,
        network_init: &[usize],
        act_in: Activation,
        act_out: Activation,
    ) -> Genetic {
        if population_size < 3 {
            panic!("Population size should be at least 3");
        }
        Genetic {
            organisms: (0..population_size)
                .map(|_| Organism {
                    fitness: 0.0,
                    nn: NeuralNetwork::new(network_init.to_vec(), act_in, act_out),
                })
                .collect(),
        }
    }
    /// Advanced evaluation function:
    /// ```rust
    /// let mut g = Genetic::new(...);
    /// g.advanced_evaluation(|organisms: &mut [Organism]| {
    ///     // write code there
    /// });
    /// ```
    pub fn advanced_evaluation(&mut self, mut fun: impl FnMut(&mut [Organism])) {
        fun(&mut self.organisms);
    }
    /// Evaluate current population in `env`. This code will evaluate organisms in parallel,if you want to evaluate without parallelism use `advanced_evaluation` function
    pub fn evaluate(&mut self, env: &mut dyn Environment) {
        use rayon::iter::*;
        self.organisms.par_iter_mut().for_each(|x| {
            env.test(x);
        })
    }
    /// Get organism with best fitness in current population
    pub fn get_champion(&self) -> &Organism {
        &self.organisms[self.fittest()[0]]
    }
    /// Returns indexes of fittest organisms
    pub fn fittest(&self) -> [usize; 3] {
        let mut fittest = [0; 3];
        for i in 0..self.organisms.len() {
            if self.organisms[fittest[0]].fitness < self.organisms[i].fitness {
                fittest[0] = i;
            }
        }
        for i in 0..self.organisms.len() {
            if i == fittest[0] {
                continue;
            }
            if self.organisms[fittest[1]].fitness < self.organisms[i].fitness {
                fittest[1] = i;
            }
        }
        for i in 0..self.organisms.len() {
            if i == fittest[0] || i == fittest[1] {
                continue;
            }
            if self.organisms[fittest[2]].fitness < self.organisms[i].fitness {
                fittest[2] = i;
            }
        }
        return fittest;
    }
    /// Create offspring by mutation and mating.
    pub fn evolve(&mut self, learn_params: &LearnParams) {
        let fittest = self.fittest();
        let parent1: Organism = self.organisms[fittest[0]].clone();
        let parent2: Organism = self.organisms[fittest[1]].clone();
        let parent3: Organism = self.organisms[fittest[2]].clone();
        /*if parent1.fitness == 0.0 && parent2.fitness == 0.0 {
            parent1 = Organism {
                fitness: 0.0,
                nn: NeuralNetwork::new(parent1.nn.layer.clone(),parent1.nn.layers[0][0].activation,parent1.nn.layers[parent1.nn.layers.len()-1][0].activation)
            };
            parent2 = Organism {
                fitness: 0.0,
                nn: NeuralNetwork::new(parent1.nn.layer.clone(),parent1.nn.layers[0][0].activation,parent1.nn.layers[parent1.nn.layers.len()-1][0].activation)
            };
        }*/
        use rayon::iter::*;
        self.organisms.par_iter_mut().for_each(|x| {
            let (m1,m2) = if random::<f64>() < 0.5 {
                (&parent1,&parent2)
            } else {
                (&parent1,&parent3)
            };
            *x = m1.mate(m1.fitness > m2.fitness, m2, learn_params);
        });
    }
}