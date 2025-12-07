# 论文笔记 - Paper Notes

这是一个基于 MkDocs + Material 主题的论文笔记网站，用于系统化整理和展示论文阅读笔记。

## 特性

- 📚 按研究领域分类组织（机器学习、计算机视觉、NLP等）
- 🔍 内置搜索功能，支持中文
- 🎨 Material 主题，现代美观，响应式设计
- 🚀 GitHub Actions 自动部署到 GitHub Pages
- ✍️ 纯 Markdown 格式，易于编写和维护
- 📱 移动端友好

## 在线访问

网站地址：`https://your-username.github.io/paper-notes/`

（请将 `your-username` 替换为您的 GitHub 用户名）

## 本地开发

### 1. 环境要求

- Python 3.7+
- pip

### 2. 安装依赖

```bash
cd paper-notes
pip install -r requirements.txt
```

### 3. 本地预览

```bash
mkdocs serve
```

然后在浏览器中访问 `http://127.0.0.1:8000`

网站会自动监听文件变化并实时刷新。

## 部署到 GitHub Pages

### 初次部署

1. **在 GitHub 创建仓库**

   创建一个名为 `paper-notes` 的新仓库（可以是私有或公开）

2. **修改配置文件**

   编辑 `mkdocs.yml`，将以下内容替换为您的信息：
   ```yaml
   site_url: https://your-username.github.io/paper-notes/
   repo_name: your-username/paper-notes
   repo_url: https://github.com/your-username/paper-notes
   ```

   编辑 `docs/about.md`，填写您的个人信息。

3. **推送代码到 GitHub**

   ```bash
   cd paper-notes
   git init
   git add .
   git commit -m "初始化论文笔记网站"
   git branch -M main
   git remote add origin https://github.com/your-username/paper-notes.git
   git push -u origin main
   ```

4. **配置 GitHub Pages**

   - 进入仓库的 **Settings** → **Pages**
   - 在 **Source** 下拉菜单中选择 `gh-pages` 分支
   - 点击 **Save**

5. **等待部署完成**

   - 查看 **Actions** 标签页，等待工作流运行完成（约 1-2 分钟）
   - 部署成功后，访问 `https://your-username.github.io/paper-notes/`

### 后续更新

每次修改后只需：

```bash
git add .
git commit -m "添加新论文笔记"
git push
```

GitHub Actions 会自动构建并部署，无需手动操作。

## 使用指南

### 添加新论文笔记

1. **选择合适的领域目录**

   进入 `docs/` 下对应的领域目录，如 `machine-learning/`、`computer-vision/`、`nlp/` 等。

2. **创建新的 Markdown 文件**

   文件名建议使用论文的简短英文名，如 `attention-is-all-you-need.md`。

3. **使用模板填写内容**

   参考现有笔记的格式，包含以下部分：

   ```markdown
   # 论文标题

   <div class="paper-meta" markdown>
   **作者**:
   **机构**:
   **会议/期刊**:
   **论文链接**:
   </div>

   <div class="paper-tags" markdown>
   <span class="paper-tag">标签1</span>
   <span class="paper-tag">标签2</span>
   </div>

   ## 📝 摘要

   ## 💡 核心思想

   ## 🏗️ 模型架构

   ## 📊 实验结果

   ## 🤔 个人笔记

   ## 🔗 相关论文
   ```

4. **更新导航配置**

   在 `mkdocs.yml` 的 `nav` 部分添加新笔记：

   ```yaml
   nav:
     - 机器学习:
         - machine-learning/index.md
         - 新论文标题: machine-learning/new-paper.md
   ```

5. **更新领域索引**

   在对应领域的 `index.md` 中添加论文链接。

### 添加新研究领域

1. **创建领域目录**

   ```bash
   mkdir docs/new-field
   ```

2. **创建索引文件**

   ```bash
   touch docs/new-field/index.md
   ```

   填写领域介绍和论文列表。

3. **更新导航**

   在 `mkdocs.yml` 的 `nav` 部分添加新领域：

   ```yaml
   nav:
     - 新领域:
         - new-field/index.md
   ```

4. **更新首页**

   在 `docs/index.md` 中添加新领域的介绍。

## 目录结构

```
paper-notes/
├── docs/                           # 文档目录
│   ├── index.md                    # 首页
│   ├── about.md                    # 关于页面
│   ├── stylesheets/                # 自定义样式
│   │   └── extra.css
│   ├── machine-learning/           # 机器学习领域
│   │   ├── index.md
│   │   └── *.md                    # 论文笔记
│   ├── computer-vision/            # 计算机视觉领域
│   │   ├── index.md
│   │   └── *.md
│   └── nlp/                        # 自然语言处理领域
│       ├── index.md
│       └── *.md
├── mkdocs.yml                      # MkDocs 配置文件
├── requirements.txt                # Python 依赖
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions 配置
├── .gitignore
└── README.md                       # 本文件
```

## 技术栈

- **[MkDocs](https://www.mkdocs.org/)** - 静态站点生成器
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - 主题
- **[GitHub Pages](https://pages.github.com/)** - 免费托管
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD 自动部署

## 常见问题

### 如何修改主题颜色？

编辑 `mkdocs.yml` 中的 `theme.palette` 部分：

```yaml
theme:
  palette:
    - scheme: default
      primary: indigo  # 改为其他颜色：blue, green, red 等
      accent: indigo
```

### 如何添加数学公式？

已配置 MathJax 支持，直接使用 LaTeX 语法：

```markdown
行内公式：$E = mc^2$

块级公式：
$$
\frac{\partial L}{\partial w} = 0
$$
```

### 如何添加代码高亮？

使用三重反引号并指定语言：

````markdown
```python
def hello():
    print("Hello, World!")
```
````

### 部署失败怎么办？

1. 检查 GitHub Actions 日志，查看具体错误信息
2. 确保 `requirements.txt` 中的依赖版本正确
3. 确认 `mkdocs.yml` 配置无误
4. 检查 Markdown 文件格式是否正确

## 许可协议

本项目采用 [MIT License](LICENSE)。

论文内容版权归原作者所有，本仓库仅提供学习笔记和个人理解。

## 致谢

- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages](https://pages.github.com/)

---

如有问题或建议，欢迎提 [Issue](https://github.com/your-username/paper-notes/issues)！
