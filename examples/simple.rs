
extern crate maver;

use maver::*;


struct XOR;

impl Environment for XOR {
    fn test(&self,org: &mut Organism) {
        let mut distance: f64;
        let nn = &mut org.nn;
        let out = nn.forward_prop(&[0.0,0.0]);
        distance = (0.0 - out[0]).powi(2);
        let out = nn.forward_prop(&[0.0,1.0]);
        distance += (1.0 - out[0]).powi(2);
        let out = nn.forward_prop(&[1.0,0.0]);
        distance += (1.0 - out[0]).powi(2);
        let out = nn.forward_prop(&[1.0,1.0]);
        distance += (0.0 - out[0]).powi(2);
        let fitness = 16.0 / (1.0 + distance);
        org.fitness = fitness;
    }
}

fn main() {
    let mut n = Genetic::new(150,&[2,5,1],Activation::Tanh,Activation::Tanh);
    let p = LearnParams {
        bias_zero_pr: 0.0,
        bias_one_pr: 0.0,
        ..LearnParams::default()
    };
    let mut champion = None;
    let mut i = 0;
    while champion.is_none() {
        n.evaluate(&mut XOR);
        if n.get_champion().fitness > 15.999 {
            champion = Some(n.get_champion().clone());
        }
        println!("Iteration: {:?},best fitness: {}",i,n.get_champion().fitness);
        n.evolve(&p);
        

        i += 1;
    }

    
    println!("{:#?}",champion.as_ref().unwrap());
    println!("{:?}",champion.as_mut().unwrap().nn.forward_prop(&[0.0,0.0]));
    println!("{:?}",champion.as_mut().unwrap().nn.forward_prop(&[1.0,0.0]));
    println!("{:?}",champion.as_mut().unwrap().nn.forward_prop(&[1.0,1.0]));
    println!("{:?}",champion.as_mut().unwrap().nn.forward_prop(&[0.0,1.0]));
}
