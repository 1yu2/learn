# 附录: 学习资源

## 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| The Rust Book | https://doc.rust-lang.org/book/ | Rust 圣经，入门必读 |
| Rust by Example | https://doc.rust-lang.org/rust-by-example/ | 通过实例学习 Rust |
| Rustlings | https://github.com/rust-lang/rustlings/ | 交互式命令行练习 |
| 标准库文档 | https://doc.rust-lang.org/std/ | 标准库 API 参考 |
| Cargo 手册 | https://doc.rust-lang.org/cargo/ | 包管理器文档 |
| Rust Reference | https://doc.rust-lang.org/reference/ | 语言参考 |
| Rustonomicon | https://doc.rust-lang.org/nomicon/ | Unsafe Rust 指南 |

## 社区

- 用户论坛: https://users.rust-lang.org
- Discord: https://discord.gg/rust-lang
- 中文社区: https://rustcc.cn

## 其他推荐

- 布朗大学互动版 Rust Book: https://rust-book.cs.brown.edu (带测验、可视化)
- Rust design patterns: https://rust-unofficial.github.io/patterns/

## 关键概念速查

| 概念 | 说明 |
|------|------|
| 所有权 (Ownership) | 每个值有唯一的所有者，离开作用域时释放 |
| 借用 (Borrowing) | 通过引用临时使用值而不获取所有权 |
| 生命周期 (Lifetime) | 编译器验证引用有效性 |
| Move 语义 | 赋值/传参会转移所有权 (堆数据) |
| Copy 语义 | 栈数据赋值时自动复制 |
| Trait | 类似接口，定义共享行为 |
| match | 穷尽模式匹配 |
| Result / Option | 无 null/异常，显式处理 |
| unsafe | 绕过部分安全检查 (谨慎使用) |
| Cargo | 官方构建系统与包管理器 |
