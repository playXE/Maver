extern crate maver;

use maver::*;

struct Add;

impl Environment for Add {
    fn test(&self, org: &mut Organism) {
        let mut distance: f64 = 0.0;
        let nn = &mut org.nn;
        let x = rand::random::<i16>() % 255;
        let y = rand::random::<i16>() % 255;
        let out = nn.forward_prop(&[x as f64, y as f64]);
        distance += (x.wrapping_add(y) as f64 - out[0]).powi(2);
        let x = rand::random::<i16>();
        let y = rand::random::<i16>();
        let out = nn.forward_prop(&[x as f64, y as f64]);
        distance += ((x.wrapping_add(y)) as f64 - out[0]).powi(2);
        let x = rand::random::<i16>();
        let y = rand::random::<i16>();
        let out = nn.forward_prop(&[x as f64, y as f64]);
        distance += (x.wrapping_add(y) as f64 - out[0]).powi(2);
        let x = rand::random::<i16>();
        let y = rand::random::<i16>();
        let out = nn.forward_prop(&[x as f64, y as f64]);
        distance += (x.wrapping_add(y) as f64 - out[0]).powi(2);
        let fitness = 16.0 / (1.0 + distance);
        org.fitness = fitness;
    }
}

struct BitAdd;

fn nn_input(x: i8, y: i8) -> [f64; 16] {
    let mut arr = [0.0; 16];
    let bits = format!("{:b}", x)
        .chars()
        .map(|x| if x == '1' { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    arr[0] = bits[0];
    arr[1] = bits[1];
    arr[2] = bits[2];
    arr[3] = bits[3];
    arr[4] = bits[4];
    arr[5] = bits[5];
    arr[6] = bits[6];
    arr[7] = bits[7];
    let bits = format!("{:b}", y)
        .chars()
        .map(|x| if x == '1' { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();

    arr[8] = bits[0];
    arr[9] = bits[1];
    arr[10] = bits[2];
    arr[11] = bits[3];
    arr[12] = bits[4];
    arr[13] = bits[5];
    arr[14] = bits[6];
    arr[15] = bits[7];
    arr
}

impl Environment for BitAdd {
    fn test(&self, org: &mut Organism) {
        let mut distance: f64 = 0.0;
        let nn = &mut org.nn;
        let x = rand::random::<i8>();
        let y = rand::random::<i8>();
        let out = nn.forward_prop(&nn_input(x, y));
        distance += ((x + y) as f64
            - i8::from_str_radix(
                &out.iter()
                    .map(|x| if x.floor() == 1.0 { '1' } else { '0' })
                    .collect::<String>(),
                2,
            )
            .unwrap() as f64)
            .powi(2);
        let x = rand::random::<i8>();
        let y = rand::random::<i8>();
        let out = nn.forward_prop(&nn_input(x, y));
        distance += ((x + y) as f64
            - i8::from_str_radix(
                &out.iter()
                    .map(|x| if x.floor() == 1.0 { '1' } else { '0' })
                    .collect::<String>(),
                2,
            )
            .unwrap() as f64)
            .powi(2);
        let x = rand::random::<i8>();
        let y = rand::random::<i8>();
        let out = nn.forward_prop(&nn_input(x, y));
        distance += ((x + y) as f64
            - i8::from_str_radix(
                &out.iter()
                    .map(|x| if x.floor() == 1.0 { '1' } else { '0' })
                    .collect::<String>(),
                2,
            )
            .unwrap() as f64)
            .powi(2);
        let x = rand::random::<i8>();
        let y = rand::random::<i8>();
        let out = nn.forward_prop(&nn_input(x, y));
        distance += ((x + y) as f64
            - i8::from_str_radix(
                &out.iter()
                    .map(|x| if x.floor() == 1.0 { '1' } else { '0' })
                    .collect::<String>(),
                2,
            )
            .unwrap() as f64)
            .powi(2);

        let fitness = 16.0 / (1.0 + distance);
        org.fitness = fitness;
    }
}

fn main() {
    let mut n = Genetic::new(400, &[2, 1], Activation::Linear, Activation::Linear);
    let p = LearnParams {
        bias_zero_pr: 0.0,
        bias_one_pr: 0.0,

        ..LearnParams::default()
    };
    let mut champion = None;
    let mut i = 0;

    while champion.is_none() {
        n.evaluate(&mut Add);
        if n.get_champion().fitness > 15.6 {
            champion = Some(n.get_champion().clone());
        }
        println!(
            "Iteration: {:?},best fitness: {}",
            i,
            n.get_champion().fitness
        );
        n.evolve(&p);

        i += 1;
    }

    println!("{:#?}", champion.as_ref().unwrap());
    println!(
        "{:?}",
        champion.as_mut().unwrap().nn.forward_prop(&[-244.0, 1.0])[0].floor()
    );
    println!(
        "{:?}",
        champion.as_mut().unwrap().nn.forward_prop(&[-244.0, -1.0])[0].floor()
    );
}
