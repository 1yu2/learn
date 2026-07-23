use std::collections::HashMap;

fn main() {
    // ========== Vector ==========
    println!("=== Vector ===");
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);

    let v2 = vec![10, 20, 30, 40, 50];

    let third = &v2[2];
    println!("第三个元素: {third}");

    match v2.get(10) {
        Some(value) => println!("第11个元素: {value}"),
        None => println!("没有第11个元素"),
    }

    for i in &v2 {
        print!("{i} ");
    }
    println!();

    for i in &mut v {
        *i += 50;
    }
    println!("v 加50后: {v:?}");

    // ========== 字符串 ==========
    println!("\n=== 字符串 ===");
    let mut s = String::new();
    s.push_str("hello");
    s.push('!');
    println!("s = {s}");

    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = format!("{s1}{s2}");
    println!("s3 = {s3}");

    // ========== HashMap ==========
    println!("\n=== HashMap ===");
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    let team = String::from("Blue");
    let score = scores.get(&team).copied().unwrap_or(0);
    println!("{team}: {score}");

    scores.entry(String::from("Blue")).or_insert(100);
    scores.entry(String::from("Red")).or_insert(30);

    for (key, value) in &scores {
        println!("{key}: {value}");
    }

    // ========== 单词计数示例 ==========
    println!("\n=== 单词计数 ===");
    let text = "hello world hello rust world hello";
    let mut map = HashMap::new();
    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;
    }
    println!("{map:?}");
}
