# Go 学习计划

本仓库用于按阶段系统学习 [Go](https://go.dev/) 编程语言。Go 是由 Google 开发的开源编程语言，以简洁、高效和强大的并发支持著称。

资料以 Go 官方文档为主，建议边读边在仓库里沉淀可运行示例。每个阶段都要留下代码、运行记录和复盘笔记。

## 学习目标

- 掌握 Go 的基础语法、类型系统和控制流。
- 理解 Go 的函数（Functions）、方法（Methods）和接口（Interfaces）。
- 学会 Go 的包（Packages）和模块（Modules）管理。
- 掌握 Go 的结构体（Structs）、嵌入（Embedding）和组合（Composition）。
- 深入理解 Go 的并发模型：Goroutines、Channels、Mutex 和 Select。
- 掌握 Go 的错误处理（Error Handling）和 Panic/Recover 机制。
- 学会 Go 的测试（Testing）、基准测试（Benchmark）和模糊测试（Fuzzing）。
- 能用 Go 构建 Web 服务（net/http、Gin）。
- 能用 Go 操作数据库（database/sql）。
- 能用 Go 构建命令行工具（CLI）。
- 理解 Go 的工具链：go build、go test、go mod、go vet、pprof 等。

## 环境准备

建议安装最新版本的 Go：

- [下载 Go](https://go.dev/dl/)
- [安装指南](https://go.dev/doc/install)

验证安装：

```bash
go version
```

## 项目代码规划

仓库采用"阶段示例 + 共享模块 + 最终综合项目"的方式推进：

- `examples/`：按学习阶段存放可独立运行的最小示例。
- `notes/`：记录概念理解、错误排查和复盘。
- `go.mod`：Go 模块定义文件。
- `tmp/`：本地运行时产物，不提交到 Git。

## 仓库结构

```text
.
├── README.md
├── go.mod
├── .gitignore
├── notes/
│   └── concepts.md
├── examples/
│   ├── 01_basics/            # 基础语法：变量、类型、控制流
│   ├── 02_functions/         # 函数与方法
│   ├── 03_packages_modules/  # 包与模块管理
│   ├── 04_structs_interfaces/# 结构体、嵌入与接口
│   ├── 05_concurrency/       # 并发：Goroutine、Channel、Select
│   ├── 06_error_handling/    # 错误处理与 Panic/Recover
│   ├── 07_testing/           # 测试、基准测试与模糊测试
│   ├── 08_web_service/       # Web 服务：net/http、Gin
│   ├── 09_database/          # 数据库：database/sql
│   └── 10_cli_app/           # CLI 应用
└── tmp/
```

## 阶段 0：理解 Go 全貌

目标：先建立地图，不急着写复杂代码。

阅读：

- [A Tour of Go](https://go.dev/tour/) — 交互式入门，分四部分
- [Effective Go](https://go.dev/doc/effective_go) — 写出地道的 Go 代码
- [How to Write Go Code](https://go.dev/doc/code) — 了解 Go 的项目管理方式

重点理解：

- Go 是静态类型、编译型语言，但语法简洁像动态语言。
- Go 没有类（Class），用结构体（Struct）+ 方法（Method）+ 接口（Interface）实现面向对象。
- Go 用 Goroutine 和 Channel 实现并发，"不要通过共享内存来通信，而要通过通信来共享内存"。
- Go 的错误处理通过返回值传递 error，没有 try-catch 异常机制。
- Go 的包管理从 $GOPATH 演进到 Go Modules（go.mod）。

产出：

- 在 `notes/concepts.md` 记录核心概念。
- 画出一张简单关系图：Goroutine -> Channel -> Select -> 并发模式。

## 阶段 1：基础语法

目标：掌握变量、类型、控制流、数组、切片、映射等基础。

阅读：

- [A Tour of Go - Basics](https://go.dev/tour/basics/)
- [Go Spec - Lexical elements](https://go.dev/ref/spec#Lexical_elements)

练习：

- 在 `examples/01_basics/` 创建文件，练习：
  - 变量声明（var、:=）
  - 基本类型（int、float、string、bool）
  - 零值（Zero values）
  - 类型转换
  - 常量（const）
  - for 循环（Go 只有 for）
  - if/else、switch
  - defer
  - 数组（array）、切片（slice）、映射（map）
  - range

检查点：

- 能解释 Go 中 := 和 var 的区别。
- 能说明数组和切片的差异及使用场景。
- 能写出 defer 的多种用法。

## 阶段 2：函数与方法

目标：理解函数声明、多返回值、变参、闭包、方法。

阅读：

- [A Tour of Go - Functions](https://go.dev/tour/basics/4)
- [Go Spec - Function declarations](https://go.dev/ref/spec#Function_declarations)
- [Effective Go - Functions](https://go.dev/doc/effective_go#functions)

重点理解：

- Go 函数支持多返回值，常用于返回 (result, error)。
- 函数是一等公民：可以赋值给变量、作为参数传递、作为返回值。
- 闭包（Closure）可以捕获外部变量。
- 变参函数（Variadic Functions）使用 ... 语法。
- 方法（Method）是带接收者（Receiver）的函数。

练习：

- 在 `examples/02_functions/` 创建文件，练习：
  - 多返回值
  - 命名返回值
  - 错误返回模式
  - 闭包
  - 函数作为参数
  - 变参函数
  - 值接收者与指针接收者的方法

检查点：

- 能解释何时用值接收者，何时用指针接收者。
- 能写出一个简单的高阶函数（接受函数作为参数）。

## 阶段 3：包与模块管理

目标：理解 Package、Module、导入路径、依赖管理。

阅读：

- [How to Write Go Code](https://go.dev/doc/code)
- [Managing dependencies](https://go.dev/doc/modules/managing-dependencies)
- [Go Modules Reference](https://go.dev/ref/mod)
- [Using Go Modules](https://go.dev/blog/using-go-modules)

重点理解：

- 每个 Go 文件都属于一个 package，package 是代码组织的基本单元。
- Go Modules 是依赖管理方案，go.mod 记录模块路径和依赖。
- go.sum 保存依赖的校验和。
- 大写字母开头的标识符是导出的（Exported），小写字母开头的是未导出的。
- import 路径可以是标准库、第三方模块或项目内部路径。

练习：

- 在 `examples/03_packages_modules/` 创建一个简单的多包项目。
- 使用 go mod init 初始化模块。
- 引入一个第三方包（如 `github.com/google/uuid`），运行 go mod tidy。
- 理解并实践 internal 包的使用（`go 1.4` 起约定 internal 目录不可被外部导入）。

检查点：

- 能解释 go.mod 和 go.sum 的作用。
- 能用 go get 添加依赖，go mod tidy 清理依赖。
- 能说明导出规则（首字母大写）。

## 阶段 4：结构体、嵌入与接口

目标：掌握 Go 面向对象编程的方式。

阅读：

- [A Tour of Go - Structs](https://go.dev/tour/moretypes/2)
- [A Tour of Go - Methods and Interfaces](https://go.dev/tour/methods/)
- [Effective Go - Interfaces](https://go.dev/doc/effective_go#interfaces_and_types)
- [Go Spec - Interface types](https://go.dev/ref/spec#Interface_types)

重点理解：

- 结构体（Struct）是字段的集合，没有类的概念。
- 嵌入（Embedding）实现组合优于继承。
- 接口（Interface）是方法签名的集合，鸭子类型（Duck Typing）。
- Go 接口是隐式实现的（不需要显式声明 implements）。
- 空接口 `interface{}`（或 `any`）可以表示任意类型。
- 类型断言（Type Assertion）和类型开关（Type Switch）。

练习：

- 在 `examples/04_structs_interfaces/` 创建文件，练习：
  - 定义结构体和嵌入
  - 为结构体实现方法
  - 定义接口并实现
  - 多态：通过接口参数实现通用行为
  - 类型断言和类型开关
  - 使用 `io.Reader` 和 `io.Writer` 接口

检查点：

- 能解释为什么 Go 不通过 extends 而通过嵌入实现代码复用。
- 能说明接口的隐式实现带来什么好处。
- 能准确描述空接口 `any` 的使用限制。

## 阶段 5：并发编程

目标：掌握 Goroutine、Channel、Select、Mutex、WaitGroup。

阅读：

- [A Tour of Go - Concurrency](https://go.dev/tour/concurrency/)
- [Go Concurrency Patterns](https://www.youtube.com/watch?v=f6kdp27TYZs)
- [Advanced Go Concurrency Patterns](https://www.youtube.com/watch?v=QDDwwePbDtw)
- [Share Memory by Communicating](https://go.dev/doc/codewalk/sharemem/)
- [Go Blog - Pipelines and cancellation](https://go.dev/blog/pipelines)

重点理解：

- Goroutine 是轻量级线程，由 Go 运行时调度。
- Channel 是 Goroutine 之间通信的管道。
- 无缓冲 Channel 是同步的，有缓冲 Channel 是异步的。
- Select 用于多路 Channel 操作。
- sync.Mutex、sync.RWMutex 提供传统锁机制。
- sync.WaitGroup 用于等待一组 Goroutine 完成。
- context.Context 用于传递取消信号、超时和请求范围的值。

练习：

- 在 `examples/05_concurrency/` 创建文件，练习：
  - 启动 Goroutine
  - 无缓冲 Channel 通信
  - 有缓冲 Channel 通信
  - Select 多路复用
  - Mutex 互斥锁
  - WaitGroup 同步
  - Context 取消和超时
  - 简单的并发模式：Pipeline、Fan-out/Fan-in

检查点：

- 能用 Channel 在 Goroutine 之间安全传递数据。
- 能说明无缓冲和有缓冲 Channel 的行为差异。
- 能使用 Context 实现超时控制。
- 能避免常见的并发陷阱（Goroutine 泄漏、死锁）。

## 阶段 6：错误处理与 Panic/Recover

目标：掌握 Go 的错误处理哲学。

阅读：

- [Effective Go - Errors](https://go.dev/doc/effective_go#errors)
- [Go Blog - Error Handling and Go](https://go.dev/blog/error-handling-and-go)
- [Go Blog - Defer, Panic, and Recover](https://go.dev/blog/defer-panic-and-recover)
- [Working with Errors in Go 1.13](https://go.dev/blog/go1.13-errors)

重点理解：

- Go 通过返回值传递 error，调用者负责检查。
- errors.New() 和 fmt.Errorf() 创建错误。
- Go 1.13+ 的错误包装：fmt.Errorf("%w", err) 和 errors.Unwrap()。
- errors.Is() 和 errors.As() 用于错误检查和类型断言。
- panic 是严重错误，用于不可恢复的情况。
- recover 在 defer 中捕获 panic。
- 不要让 panic 跨越包边界。

练习：

- 在 `examples/06_error_handling/` 创建文件，练习：
  - 检查和处理 error
  - 创建自定义错误类型
  - 错误包装和解包
  - 使用 errors.Is 和 errors.As
  - defer + recover 捕获 panic

检查点：

- 能区分使用 error 和 panic 的场景。
- 能正确包装和展开错误链。
- 能说明 errors.Is 和 errors.As 的区别。

## 阶段 7：测试

目标：掌握 Go 的测试、基准测试和模糊测试。

阅读：

- [How to Write Go Code - Testing](https://go.dev/doc/code#Testing)
- [Testing package](https://pkg.go.dev/testing)
- [Tutorial: Getting started with fuzzing](https://go.dev/doc/tutorial/fuzz/)
- [Coverage for Go applications](https://go.dev/doc/build-cover)

重点理解：

- 测试文件以 `_test.go` 结尾。
- 测试函数以 `Test` 开头，签名 `func(t *testing.T)`。
- 基准测试以 `Benchmark` 开头，签名 `func(b *testing.B)`。
- 子测试（Subtests）使用 `t.Run()` 组织。
- 表驱动测试（Table-Driven Tests）是 Go 社区的惯用模式。
- `t.Parallel()` 标记测试可以并行运行。
- 模糊测试（Fuzzing）自动生成随机输入。
- `go test -cover` 查看代码覆盖率。

练习：

- 在 `examples/07_testing/` 为之前的代码编写测试。
  - 表驱动测试
  - 子测试
  - 基准测试
  - 模拟（Mock）外部依赖
  - 使用 `go test -cover` 查看覆盖率

检查点：

- 能写出规范的表驱动测试。
- 能区分 Test 和 Benchmark 的使用场景。
- 能使用 Mock 隔离外部依赖。

## 阶段 8：Web 服务

目标：能用 Go 构建 RESTful API。

阅读：

- [Tutorial: Developing a RESTful API with Go and Gin](https://go.dev/doc/tutorial/web-service-gin)
- [Writing Web Applications](https://go.dev/doc/articles/wiki/)
- [net/http package](https://pkg.go.dev/net/http)

重点理解：

- `net/http` 标准库提供 HTTP 客户端和服务器。
- `http.Handler` 和 `http.HandlerFunc` 是核心接口。
- 路由（Routing）、中间件（Middleware）和处理函数。
- JSON 编解码：`encoding/json`。
- Gin 是流行的第三方 Web 框架。

练习：

- 在 `examples/08_web_service/` 创建文件，练习：
  - 用 net/http 构建简单 HTTP 服务器
  - GET / POST / PUT / DELETE 处理
  - JSON 请求和响应
  - 中间件（日志、认证）
  - 用 Gin 重写同样功能

检查点：

- 能用 net/http 构建基本的 RESTful API。
- 理解中间件的实现原理。
- 能比较 net/http 和 Gin 的差异。

## 阶段 9：数据库操作

目标：用 Go 操作关系型数据库。

阅读：

- [Tutorial: Accessing a relational database](https://go.dev/doc/tutorial/database-access)
- [Accessing relational databases](https://go.dev/doc/database/)
- [database/sql package](https://pkg.go.dev/database/sql)

重点理解：

- `database/sql` 提供通用数据库接口。
- 需要导入具体数据库驱动（如 `github.com/mattn/go-sqlite3`）。
- `sql.DB` 管理连接池。
- Query、QueryRow、Exec 的区别。
- 预编译语句（Prepared Statements）。
- 事务（Transactions）。
- SQL 注入防护。
- Context 超时和取消。

练习：

- 在 `examples/09_database/` 创建文件，练习：
  - 连接 SQLite
  - 创建表
  - CRUD 操作
  - 事务
  - 预编译语句
  - Context 超时

检查点：

- 能用 database/sql 完成基本 CRUD。
- 能正确处理事务的提交和回滚。
- 能防止 SQL 注入。

## 阶段 10：CLI 应用

目标：用 Go 构建命令行工具。

阅读：

- [flag package](https://pkg.go.dev/flag)
- [Cobra](https://github.com/spf13/cobra) — 流行的 CLI 框架
- [Building an awesome CLI app in Go](https://spf13.com/presentation/building-an-awesome-cli-app-in-go-oscon/)

练习：

- 在 `examples/10_cli_app/` 创建文件，练习：
  - 使用 flag 包解析命令行参数
  - 构建简单 CLI 工具（如文件操作、数据转换）
  - 使用 Cobra 构建带子命令的 CLI

检查点：

- 能用 flag 标准库处理命令行参数。
- 能使用 Cobra 构建多子命令的 CLI 工具。

## 综合项目建议

选择一个足够小但覆盖核心能力的项目：

1. 文件处理工具：读取、转换、统计文件内容。
2. 简单 Web 服务：提供 CRUD API + 数据库存储。
3. 并发爬虫：并发获取多个 URL 的内容并汇总。
4. CLI 代办事项（Todo）工具。

最低验收标准：

- 使用至少一个自定义包和模块。
- 有测试覆盖。
- 有 error 处理。
- 有并发应用（Goroutine + Channel 或 Context）。
- 代码符合 Effective Go 规范。

## 学习节奏

建议每个阶段都按这个循环推进：

1. 阅读对应官方文档和 A Tour of Go。
2. 写一个最小可运行示例。
3. 记录运行命令、输出截图或关键日志。
4. 写下一个失败案例和修正方式。
5. 提交一次 Git commit。

推荐 commit 粒度：

```text
docs: add go core concept notes
feat: add go basics examples
feat: add go functions and methods examples
feat: add go packages and modules examples
feat: add go structs and interfaces examples
feat: add go concurrency examples
feat: add go error handling examples
feat: add go testing examples
feat: add go web service examples
feat: add go database examples
feat: add go cli app examples
```

## 官方文档入口

- [Go 官网](https://go.dev/)
- [A Tour of Go（交互式教程）](https://go.dev/tour/)
- [Go by Example（代码示例）](https://gobyexample.com/)
- [Effective Go（最佳实践）](https://go.dev/doc/effective_go)
- [Go Language Specification（语言规范）](https://go.dev/ref/spec)
- [Standard Library（标准库文档）](https://pkg.go.dev/std)
- [Go User Manual（用户手册）](https://go.dev/doc/)
- [Go Modules Reference（模块参考）](https://go.dev/ref/mod)
- [Go Memory Model（内存模型）](https://go.dev/ref/mem)
- [Go Blog（官方博客）](https://go.dev/blog/)
- [Go Wiki（社区维基）](https://go.dev/wiki/)

## 当前进度

- [ ] 阶段 0：理解 Go 全貌
- [ ] 阶段 1：基础语法
- [ ] 阶段 2：函数与方法
- [ ] 阶段 3：包与模块管理
- [ ] 阶段 4：结构体、嵌入与接口
- [ ] 阶段 5：并发编程
- [ ] 阶段 6：错误处理与 Panic/Recover
- [ ] 阶段 7：测试
- [ ] 阶段 8：Web 服务
- [ ] 阶段 9：数据库操作
- [ ] 阶段 10：CLI 应用
