# 第一章 快速入门

## 安装 Rust

通过 `rustup` 安装 Rust 工具链:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# 验证安装
rustc --version
```

macOS 还需要安装 C 编译器:

```bash
xcode-select --install
```

## Hello, World!

```rust
fn main() {
    println!("Hello, world!");
}
```

编译运行:

```bash
rustc main.rs
./main
```

## Cargo — Rust 的构建系统与包管理器

```bash
# 创建新项目
cargo new hello_cargo
cd hello_cargo

# 构建项目
cargo build

# 构建并运行
cargo run

# 快速检查代码是否能编译 (不生成可执行文件)
cargo check

# 发布构建 (优化)
cargo build --release
```

Cargo.toml 示例:

```toml
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2024"

[dependencies]
rand = "0.8.5"
```

## 更新 Rust

```bash
rustup update          # 更新 Rust
rustup self uninstall  # 卸载 Rust
rustup doc             # 打开本地文档
```
