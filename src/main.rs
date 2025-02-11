mod print;
fn main() {
    print::run();

    println!("Number : {}", 12);

    println!("{} is from {}", "Mustafa", "Bara");

    // Positional Arguments
    println!("{0} is from {1} and {0} likes to {2}", "Mustafa", "Mars", "Code");

    // Named Arguments
    println!("{name} likes to play {activity}", name="He", activity="Basketball");

    // Placeholder traits
    println!("Binary: {:b} Hex: {:x} Octal: {:o}", 10, 10 , 10);

    // Placeholder for debug trait
    println!("{:?}", (12, true, "hello"));
}
