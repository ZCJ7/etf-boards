# etfirst 安装指南

依次执行第 1 → 第 3 步。

## 前置条件

- Python 3.8+
- pip
- API Key（联系管理员获取）

## 第 1 步 安装

> **⚠️ 如果之前安装过旧版 `etfapp-cli`，必须先完成「清理旧版 etfapp-cli」章节的所有步骤，并删除旧版 SKILL.md，再执行安装。**

下载安装包：

```
https://www.nffund.com/wxfiles/miniapp/openu/static/assets/etfirst/0.2.3/etfirst-0.2.3.tar.gz
```

安装：

```shell
pip install etfirst-0.2.3.tar.gz
```

验证：
```shell
etfirst --help
```

## 第 2 步 登录

向用户索取 API Key 后，替换 `<YOUR_API_KEY>` 执行：

```shell
etfirst auth login --api-key <YOUR_API_KEY>
```

API Key 持久化保存位置：

- 默认路径：`~/.cli_anything/etfapp/config.json`（与 session.json 分离存储）
- 自定义路径：设置环境变量 `ETFAPP_CLI_CONFIG=<绝对路径>`
- 清除时机：仅 `etfirst auth logout` 或手动删除该文件才会清除；CLI 卸载 / 升级 / session 失效均不影响
- 一台机器仅需登录一次

## 第 3 步 验证

```shell
etfirst config
etfirst --json index-base clas --type 1
etfirst --json index-base list-etf --type 2 --page-no 1 --page-size 3
```

## 升级

```shell
pip uninstall etfirst
pip install etfirst-<新版本>.tar.gz
```

升级不影响 API Key，无需重新登录。

## 清理旧版 etfapp-cli

如果之前安装过旧版 `etfapp-cli`，需先卸载以避免命令冲突：

```shell
pip uninstall etfapp-cli
```

然后更新 AI Agent 中的 skill 文件：

**第一步：定位旧 SKILL.md**

旧版 SKILL.md 通常放在你向 Agent 配置的 skill 目录中，例如：

| 框架 | 常见路径示例 |
|------|--------------|
| Qoder / Qoder Worker | `.qoder/skills/<包名>/SKILL.md` |
| OpenClaw | `~/.openclaw/skills/<包名>/SKILL.md` |
| 其他框架 | 参考该框架的 skills 目录配置 |

可以全文搜索确认旧文件位置（内容包含 `etfapp-cli`）：

```shell
# macOS / Linux
grep -rl "etfapp-cli" ~/  --include="SKILL.md" 2>/dev/null

# Windows PowerShell
Get-ChildItem -Path $HOME -Recurse -Filter "SKILL.md" -ErrorAction SilentlyContinue | Select-String "etfapp-cli" | Select-Object -ExpandProperty Path
```

将定位到的旧版 SKILL.md 直接删除即可。