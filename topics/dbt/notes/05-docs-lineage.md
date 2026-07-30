# 05 文档与血缘（Docs & Lineage）

## 为什么文档要写进代码里

数据项目最常见的文档灾难：wiki 里一份口径、SQL 里另一份口径，三个月后没人知道哪个对。dbt 的思路是**文档即代码**：description 写在 model 旁边的 yml 里，跟着 SQL 一起进 Git、一起评审、一起演进，然后一键生成可浏览的文档站和血缘图。

## description：写在 schema.yml 里

model 和列都可以加 `description`，和 tests 写在同一个 yml 文件里：

```yaml
version: 2

models:
  - name: support_kpi_mart
    description: "支持团队 KPI 汇总 mart（按类别）：工单量、解决时长、SLA 达标率。"
    columns:
      - name: category
        description: "工单类别"
      - name: sla_compliance_rate
        description: "SLA 达标率（0-1 之间的小数）"
```

source 也同样可以写 description（见 `models/staging/sources.yml`）。进阶还支持 Markdown、docs block 复用大段文档，入门阶段先把每个 model 和关键列的描述写全。

> 对应 omnisupport-copilot：`analytics/models/staging/schema.yml` 里每个 stg_ model 都有 description；另外它还有一个 `metric_registry_v1.yml`——那不是 dbt 功能，而是项目自己治理指标口径的注册表（每个指标的业务定义、负责人、公式）。这提示了一个好实践：dbt 的 description 管「字段是什么」，指标口径注册表管「指标怎么算、归谁负责」。

## 生成与浏览

```bash
dbt docs generate          # 生成文档数据
dbt docs serve             # 起本地站点（默认 http://localhost:8080）
```

站点里能看：每个 model 的 SQL、description、测试情况、列信息，右下角的 **lineage graph** 可以交互式展开任意节点的上下游血缘——这就是之前 DAG 概念的可视化。

## manifest.json 与 catalog.json

`dbt docs generate`（以及 run/build）会在 `target/` 下产出两个关键文件：

- **manifest.json**：项目的完整「解析结果」——所有 model、test、source、macro 的定义、依赖关系（DAG）、配置、编译后的 SQL。每次 dbt 运行都会刷新。它是 dbt 自己调度执行的依据，也是各种第三方工具（血缘分析、成本分析）读取 dbt 项目的标准入口。
- **catalog.json**：数仓里实际的**元数据**——每张表/视图的列名、类型、统计信息。只有 `dbt docs generate` 会去数仓抓取生成它。

一句话：manifest 是「dbt 认为项目长什么样」（来自代码），catalog 是「数仓里实际长什么样」（来自数据库）。文档站把两者拼在一起展示。

## L4 自检

- [ ] 给 practice 项目所有还没有 description 的 model 和列补上描述，`dbt docs generate` 无警告通过。
- [ ] 起 `dbt docs serve`，在 lineage 图里从 `support_kpi_mart` 一路点回三个 source，确认和你心中的 DAG 一致。
- [ ] 打开 `target/manifest.json`，找到 `support_kpi_mart` 节点，指出它的 `depends_on` 写了什么。
- [ ] 说清 manifest.json 和 catalog.json 的来源差异（代码 vs 数仓）和各自的典型消费者。
- [ ] 向同事解释「文档即代码」解决了什么问题——为什么不该把口径写在独立 wiki 里。
