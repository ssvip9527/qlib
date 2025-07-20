# 版权所有 (c) Microsoft Corporation.
# 根据 MIT 许可证授权。


# QLib 文档构建配置文件，由 sphinx-quickstart 创建于 2017年9月27日 15:16:05。
#
# 此文件在当前目录设置为其包含目录时被 execfile() 执行。
#
# 注意：并非所有可能的配置项都在此自动生成的文件中出现。
#
# 所有配置项都有默认值；被注释掉的值用于展示默认值。

# 如果扩展（或 autodoc 要记录的模块）在其他目录中，
# 请在此处将这些目录添加到 sys.path。如果目录是相对于文档根目录的，
# 请像下面这样使用 os.path.abspath 使其成为绝对路径。
#
import os
import sys

# import pkg_resources  # Deprecated, see https://setuptools.pypa.io/en/latest/pkg_resources.html


# -- 通用配置 ------------------------------------------------

# 如果您的文档需要最低版本的 Sphinx，请在此处声明。
#
# needs_sphinx = '1.0'

# 在此处添加任何 Sphinx 扩展模块名称，作为字符串。它们可以是
# Sphinx 自带的扩展（命名为 'sphinx.ext.*'）或您的自定义扩展。
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

# 在此处添加包含模板的所有路径，相对于此目录。
templates_path = ["_templates"]

# 源文件的后缀。
# 您可以将多个后缀指定为字符串列表：
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# 主 toctree 文档。
master_doc = "index"


# 项目的基本信息。
project = "QLib-中文文档"
copyright = "Microsoft"
author = "Microsoft 谋决量化 翻译"

# 您正在记录的项目的版本信息，用于 |version| 和 |release| 的替换，
# 也用于构建文档中的其他各处。
#
# 简短的 X.Y 版本。
import importlib.metadata
version = importlib.metadata.version("pyqlib")
# 完整版本，包括 alpha/beta/rc 标签。
release = importlib.metadata.version("pyqlib")

# Sphinx 自动生成内容的语言。请参阅文档以获取支持的语言列表。
#
# 如果通过 gettext 目录进行内容翻译，也会用到此项。
# 通常在这些情况下从命令行设置 "language"。
language = "zh_CN"

# 相对于源目录的模式列表，用于匹配在查找源文件时要忽略的文件和目录。
# 这些模式也会影响 html_static_path 和 html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "hidden"]

# 要使用的 Pygments（语法高亮）样式名称。
pygments_style = "sphinx"

# 如果为 true，`todo` 和 `todoList` 将生成输出，否则不生成任何内容。
todo_include_todos = False

# 如果为 true，将在 :func: 等交叉引用文本后附加 '()'。
add_function_parentheses = False

# 如果为 true，当前模块名将添加到所有描述性单元标题前（如 .. function::）。
add_module_names = True

# 如果为 true，`todo` 和 `todoList` 将生成输出，否则不生成任何内容。
todo_include_todos = True


# -- HTML 输出选项 ----------------------------------------------------------

# 用于 HTML 和 HTML 帮助页面的主题。有关内置主题列表，请参阅文档。
#
html_theme = "sphinx_rtd_theme"

html_logo = "_static/img/logo/1.png"


# 主题选项是特定于主题的，用于进一步自定义主题的外观和感觉。
# 有关每个主题可用选项的列表，请参阅文档。
html_context = {
    # "display_github": False,
    # "last_updated": True,
    # "commit": True,
    # "github_user": "Microsoft",
    # "github_repo": "QLib",
    # 'github_version': 'master',
    # 'conf_py_path': '/docs/',
    "footer": "由 <a href='https://www.moujue.com'>谋决量化</a> 翻译，源码<a href='https://github.com/ssvip9527/qlib'>github</a>."
}

html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# 在此处添加包含自定义静态文件（如样式表）的任何路径，相对于此目录。
# 它们在内置静态文件之后被复制，因此名为 "default.css" 的文件将覆盖内置的 "default.css"。
# html_static_path = ['_static']

# 自定义侧边栏模板，必须是将文档名称映射到模板名称的字典。
#
# 这是 alabaster 主题所必需的
# 参考：http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",  # 需要 'show_related': True 主题选项才会显示
        "searchbox.html",
    ]
}


# -- HTMLHelp 输出选项 ------------------------------------------------------

# HTML 帮助生成器的输出文件基名。
htmlhelp_basename = "qlibdoc"


# -- LaTeX 输出选项 ---------------------------------------------------------

latex_elements = {
    # 纸张大小（'letterpaper' 或 'a4paper'）。
    #
    # 'papersize': 'letterpaper',
    # 字体大小（'10pt'、'11pt' 或 '12pt'）。
    #
    # 'pointsize': '10pt',
    # LaTeX 前言的附加内容。
    #
    # 'preamble': '',
    # LaTeX 图表（浮动）对齐方式
    #
    # 'figure_align': 'htbp',
}

# 将文档树分组到 LaTeX 文件中。元组列表
#（源起始文件，目标名称，标题，作者，文档类[howto、manual 或自定义类]）。
latex_documents = [
    (master_doc, "qlib.tex", "QLib 文档", "Microsoft", "manual"),
]


# -- 手册页输出选项 -------------------------------------------------------

# 每个手册页一个条目。元组列表
# （源起始文件，名称，描述，作者，手册部分）。
man_pages = [(master_doc, "qlib", "QLib文档", [author], 1)]

# -- Texinfo输出选项 -------------------------------------------------------

# 将文档树分组到Texinfo文件中。元组列表
# （源起始文件，目标名称，标题，作者，目录菜单条目，描述，类别）
texinfo_documents = [
    (
        master_doc,
        "QLib",
        "QLib文档",
        author,
        "QLib",
        "项目的一行描述。",
        "其他",
    ),
]


# -- Epub输出选项 ----------------------------------------------------------

# Dublin Core书目信息。
epub_title = project
epub_author = author
epub_publisher = 'https://www.moujue.com/'
epub_copyright = copyright

# 文本的唯一标识符。可以是ISBN号或项目主页。
# epub_identifier = ''

# 文本的唯一标识符。
# epub_uid = ''

# 不应打包到epub文件中的文件列表。
epub_exclude_files = ["search.html"]


autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
}
