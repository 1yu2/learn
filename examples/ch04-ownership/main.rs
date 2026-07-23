fn main() {
    // ========== Move 语义 ==========
    println!("=== Move 语义 ===");
    let s1 = String::from("hello");
    let s2 = s1;
    // println!("{s1}"); // 编译错误: s1 已被移动
    println!("s2 = {s2}");

    // ========== Clone ==========
    println!("\n=== Clone ===");
    let s3 = String::from("world");
    let s4 = s3.clone();
    println!("s3 = {s3}, s4 = {s4}");

    // ========== Copy 类型 ==========
    println!("\n=== Copy 类型 ===");
    let x = 5;
    let y = x;
    println!("x = {x}, y = {y}");

    // ========== 所有权与函数 ==========
    println!("\n=== 所有权与函数 ===");
    let s = String::from("hello");
    takes_ownership(s);

    let num = 5;
    makes_copy(num);
    println!("num 仍然可用: {num}");

    // ========== 引用与借用 ==========
    println!("\n=== 引用与借用 ===");
    let s1 = String::from("hello world");
    let len = calculate_length(&s1);
    println!("'{s1}' 的长度是 {len}");

    let mut s2 = String::from("hello");
    change(&mut s2);
    println!("修改后: {s2}");

    // 引用的规则
    let mut s3 = String::from("hello");
    let r1 = &s3;
    let r2 = &s3;
    println!("r1 = {r1}, r2 = {r2}");
    let r3 = &mut s3;
    r3.push_str(" world");
    println!("r3 = {r3}");

    // ========== 切片 ==========
    println!("\n=== 切片 ===");
    let s = String::from("hello world");
    let word = first_word(&s);
    println!("第一个单词: {word}");

    let literal = "hello world";
    let word2 = first_word(literal);
    println!("字面量第一个单词: {word2}");
}

fn takes_ownership(some_string: String) {
    println!("获取了所有权: {some_string}");
}

fn makes_copy(some_integer: i32) {
    println!("拷贝了: {some_integer}");
}

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(s: &mut String) {
    s.push_str(", Rust!");
}

fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    &s[..]
}
