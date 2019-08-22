extern crate maver;
use maver::*;

fn main () {
    let xor_sets = [
        (vec![0., 0.], vec![0.0]),
        (vec![0., 1.], vec![1.0]),
        (vec![1., 0.], vec![1.0]),
        (vec![1., 1.], vec![0.0]),
    ];
    let mut network = NeuralNetwork::new(vec![2, 2, 1],Activation::Sigmoid,Activation::Sigmoid);
    for i in 1.. {
        for &(ref input, ref output) in xor_sets.iter() {
            network.forward_set(input);
            network.backpropagate(input, output,None);
        }

        if i % 1000 == 0 {
            println!("\nIteration: {:?}", i);
            println!("eval 0,0: {:?}", network.forward_prop(&vec![0., 0.]));
            println!("eval 0,1: {:?}", network.forward_prop(&vec![0., 1.]));
            println!("eval 1,0: {:?}", network.forward_prop(&vec![1., 0.]));
            println!("eval 1,1: {:?}", network.forward_prop(&vec![1., 1.]));
        }
        if i % 1000000 == 0 {
            break
        }
    }
            println!("eval 0,0: {:?}", network.forward_prop(&vec![0., 0.]));
            println!("eval 0,1: {:?}", network.forward_prop(&vec![0., 1.]));
            println!("eval 1,0: {:?}", network.forward_prop(&vec![1., 0.]));
            println!("eval 1,1: {:?}", network.forward_prop(&vec![1., 1.]));
}