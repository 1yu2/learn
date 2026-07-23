use std::fs::{self, File};
use std::io::{self, ErrorKind, Read};

fn main() {
    // ========== panic! ==========
    // 取消注释将触发 panic:
    // panic!("主动触发 panic");

    // ========== Result 处理 ==========
    println!("=== Result 处理 ===");
    match read_file_v1("hello.txt") {
        Ok(content) => println!("文件内容: {content}"),
        Err(e) => println!("错误: {e}"),
    }

    // ========== unwrap / expect ==========
    println!("\n=== unwrap / expect ===");
    match read_file_v2("Cargo.toml") {
        Ok(content) => println!("文件内容: {content}"),
        Err(e) => println!("错误: {e}"),
    }

    // ========== ? 运算符 ==========
    println!("\n=== ? 运算符 ===");
    match read_file_v3("hello.txt") {
        Ok(content) => println!("文件内容: {content}"),
        Err(e) => println!("错误: {e}"),
    }
}

fn read_file_v1(filename: &str) -> Result<String, io::Error> {
    let file_result = File::open(filename);

    let mut file = match file_result {
        Ok(f) => f,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create(filename) {
                Ok(fc) => fc,
                Err(e) => return Err(e),
            },
            other => return Err(other),
        },
    };

    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

fn read_file_v2(filename: &str) -> Result<String, io::Error> {
    let mut content = String::new();
    File::open(filename)
        .expect("无法打开文件")
        .read_to_string(&mut content)
        .expect("无法读取文件");
    Ok(content)
}

fn read_file_v3(filename: &str) -> Result<String, io::Error> {
    fs::read_to_string(filename)
}
