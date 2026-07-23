# 第八章 常见集合

## Vector

```rust
let mut v: Vec<i32> = Vec::new();
v.push(5);
v.push(6);

let v2 = vec![1, 2, 3];

// 访问元素
let third: &i32 = &v2[2];
let third: Option<&i32> = v2.get(2);

// 遍历
for i in &v2 {
    println!("{i}");
}

for i in &mut v {
    *i += 50; // 解引用
}
```

## 字符串

```rust
let mut s = String::new();
let s1 = "initial contents".to_string();
let s2 = String::from("hello");

s.push_str("bar");
s.push('!');

let s3 = s1 + &s2; // s1 被移动，不再可用

let s4 = format!("{s2}-{s3}"); // 不获取所有权

// 遍历字符
for c in "Зд".chars() {
    println!("{c}");
}
```

## HashMap

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

let team_name = String::from("Blue");
let score = scores.get(&team_name).copied().unwrap_or(0);

// 遍历
for (key, value) in &scores {
    println!("{key}: {value}");
}

// 只在键不存在时插入
scores.entry(String::from("Blue")).or_insert(50);
```
