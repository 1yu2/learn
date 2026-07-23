fn main() {
    // ========== 变量与可变性 ==========
    println!("=== 变量 ===");

    let x = 5;
    println!("x = {x}");

    let mut y = 5;
    println!("y = {y}");
    y = 6;
    println!("y 改变后 = {y}");

    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
    println!("三小时的秒数: {THREE_HOURS_IN_SECONDS}");

    let z = 5;
    let z = z + 1;
    println!("z (遮蔽后) = {z}");

    // ========== 数据类型 ==========
    println!("\n=== 数据类型 ===");

    let a: u32 = 42;
    let b = 3.14;
    let c = true;
    let d = '🦀';
    println!("整数: {a}, 浮点数: {b}, 布尔: {c}, 字符: {d}");

    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (t0, t1, t2) = tup;
    println!("元组: {t0}, {t1}, {t2}");

    let arr = [1, 2, 3, 4, 5];
    let zeros = [0; 5];
    println!("数组: {:?}, 零数组: {:?}", arr, zeros);

    // ========== 函数 ==========
    println!("\n=== 函数 ===");
    let result = add(5, 3);
    println!("add(5, 3) = {result}");

    // ========== 控制流 ==========
    println!("\n=== 控制流 ===");

    let number = 6;
    if number % 4 == 0 {
        println!("{number} 能被 4 整除");
    } else if number % 3 == 0 {
        println!("{number} 能被 3 整除");
    } else {
        println!("{number} 不能被 4 或 3 整除");
    }

    let condition = true;
    let value = if condition { 5 } else { 6 };
    println!("if 表达式结果: {value}");

    let mut counter = 0;
    let loop_result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    println!("loop 返回值: {loop_result}");

    let mut n = 3;
    while n != 0 {
        print!("{n}!");
        n -= 1;
    }
    println!();

    for element in [10, 20, 30, 40, 50] {
        print!("{element} ");
    }
    println!();

    for i in (1..4).rev() {
        print!("{i}!");
    }
    println!("发射!");
}

fn add(x: i32, y: i32) -> i32 {
    x + y
}
