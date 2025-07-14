.PHONY: clean deepclean prerequisite dependencies lightgbm rl develop lint docs package test analysis all install dev black pylint flake8 mypy nbqa nbconvert lint build upload docs-gen
#你可以根据你的终端进行修改
SHELL := /bin/bash

########################################################################################
# 变量
########################################################################################

# 文档目标目录，将根据 readthedocs 适配到具体文件夹。
PUBLIC_DIR := $(shell [ "$$READTHEDOCS" = "True" ] && echo "$$READTHEDOCS_OUTPUT/html" || echo "public")

SO_DIR := qlib/data/_libs
SO_FILES := $(wildcard $(SO_DIR)/*.so)

ifeq ($(OS),Windows_NT)
    IS_WINDOWS = true
else
    IS_WINDOWS = false
endif

########################################################################################
# 开发环境管理
########################################################################################
# 移除常见的中间文件。
clean:
	-rm -rf \
		$(PUBLIC_DIR) \
		qlib/data/_libs/*.cpp \
		qlib/data/_libs/*.so \
		mlruns \
		public \
		build \
		.coverage \
		.mypy_cache \
		.pytest_cache \
		.ruff_cache \
		Pipfile* \
		coverage.xml \
		dist \
		release-notes.md

	find . -name '*.egg-info' -print0 | xargs -0 rm -rf
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -f
	find . -name '__pycache__' -print0 | xargs -0 rm -rf

# 移除 pre-commit 钩子、虚拟环境及中间文件。
deepclean: clean
	if command -v pre-commit > /dev/null 2>&1; then pre-commit uninstall --hook-type pre-push; fi
	if command -v pipenv >/dev/null 2>&1 && pipenv --venv >/dev/null 2>&1; then pipenv --rm; fi

# 先决条件部分
# 该代码编译两个 Cython 模块 rolling 和 expanding，使用 setuptools 和 Cython，
# 并将它们构建为可直接导入 Python 的二进制扩展模块。
# 由于 pyproject.toml 无法做到这一点，所以在这里编译。

# pywinpty 是 jupyter 在 windows 下的依赖，如果用 pip install pywinpty 安装，
# 会先下载 tar.gz 文件，然后本地编译安装，
# 这会带来一些不必要的麻烦，所以我们选择安装已编译的 whl 文件以避免麻烦。
prerequisite:
	@if [ -n "$(SO_FILES)" ]; then \
		echo "已存在共享库文件，跳过构建。"; \
	else \
		echo "未找到共享库文件，正在构建..."; \
		pip install --upgrade setuptools wheel; \
		python -m pip install cython numpy; \
		python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; extensions = [Extension('qlib.data._libs.rolling', ['qlib/data/_libs/rolling.pyx'], language='c++', include_dirs=[numpy.get_include()]), Extension('qlib.data._libs.expanding', ['qlib/data/_libs/expanding.pyx'], language='c++', include_dirs=[numpy.get_include()])]; setup(ext_modules=cythonize(extensions, language_level='3'), script_args=['build_ext', '--inplace'])"; \
	fi

	@if [ "$(IS_WINDOWS)" = "true" ]; then \
		python -m pip install pywinpty --only-binary=:all:; \
	fi

# 以可编辑模式安装包。
dependencies:
	python -m pip install -e .

lightgbm:
	python -m pip install lightgbm --prefer-binary

rl:
	python -m pip install -e .[rl]

develop:
	python -m pip install -e .[dev]

lint:
	python -m pip install -e .[lint]

docs:
	python -m pip install -e .[docs]

package:
	python -m pip install -e .[package]

test:
	python -m pip install -e .[test]

analysis:
	python -m pip install -e .[analysis]

all:
	python -m pip install -e .[pywinpty,dev,lint,docs,package,test,analysis,rl]

install: prerequisite dependencies

dev: prerequisite all

########################################################################################
# 代码检查和预提交
########################################################################################

# 使用 black 检查代码格式。
black:
	black . -l 120 --check --diff

# 使用 pylint 检查代码文件夹。
# TODO: 这些问题我们将在未来解决，重要的有：W0221、W0223、W0237、E1102
#  C0103: invalid-name
#  C0209: consider-using-f-string
#  R0402: consider-using-from-import
#  R1705: no-else-return
#  R1710: inconsistent-return-statements
#  R1725: super-with-arguments
#  R1735: use-dict-literal
#  W0102: dangerous-default-value
#  W0212: protected-access
#  W0221: arguments-differ
#  W0223: abstract-method
#  W0231: super-init-not-called
#  W0237: arguments-renamed
#  W0612: unused-variable
#  W0621: redefined-outer-name
#  W0622: redefined-builtin
#  FIXME: specify exception type
#  W0703: broad-except
#  W1309: f-string-without-interpolation
#  E1102: not-callable
#  E1136: unsubscriptable-object
#  W4904: deprecated-class
#  R0917: too-many-positional-arguments
#  E1123: unexpected-keyword-arg
# 禁用错误参考：https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html
# 我们使用 sys.setrecursionlimit(2000) 增大递归深度以保证 pylint 正常工作（默认递归深度为 1000）。
# 参数参考：https://github.com/PyCQA/pylint/issues/4577#issuecomment-1000245962
pylint:
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,W4904,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1730,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' qlib --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)"
	pylint --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,E1123,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0246,W0612,W0621,W0622,W0703,W1309,E1102,E1136 --const-rgx='[a-z_][a-z0-9_]{2,30}' scripts --init-hook="import astroid; astroid.context.InferenceContext.max_inferred = 500; import sys; sys.setrecursionlimit(2000)"

# 使用 flake8 检查代码。
# 忽略以下 flake8 错误码：
# E501 行过长
#  说明：我们已用 black 限制每行长度为 120。
# F541 f-string 缺少占位符
#  说明：pylint 检查时也做了同样处理。
# E266 块注释 # 过多
#  说明：为提升代码可读性，使用了大量 "#"。
#         该错误主要出现在：
#             qlib/backtest/executor.py
#             qlib/data/ops.py
#             qlib/utils/__init__.py
# E402 模块级 import 不在文件顶部
#  说明：有时模块级 import 无法放在文件顶部。
# W503 二元运算符前换行
#  说明：black 格式化时，算术过长需换行。
# E731 不要赋值 lambda 表达式，建议用 def
#  说明：限制 lambda 表达式，但有时必须用。
# E203 ":" 前有空格
#  说明：":" 前有空格无法通过 black 检查。
flake8:
	flake8 --ignore=E501,F541,E266,E402,W503,E731,E203 --per-file-ignores="__init__.py:F401,F403" qlib

# 使用 mypy 检查代码。
# https://github.com/python/mypy/issues/10600
mypy:
	mypy qlib --install-types --non-interactive
	mypy qlib --verbose

# 使用 nbqa 检查 ipynb。
nbqa:
	nbqa black . -l 120 --check --diff
	nbqa pylint . --disable=C0104,C0114,C0115,C0116,C0301,C0302,C0411,C0413,C1802,R0401,R0801,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1720,W0105,W0123,W0201,W0511,W0613,W1113,W1514,E0401,E1121,C0103,C0209,R0402,R1705,R1710,R1725,R1735,W0102,W0212,W0221,W0223,W0231,W0237,W0612,W0621,W0622,W0703,W1309,E1102,E1136,W0719,W0104,W0404,C0412,W0611,C0410 --const-rgx='[a-z_][a-z0-9_]{2,30}'

# 使用 nbconvert 检查 ipynb（数据下载后运行）。
# TODO: 未来添加更多 ipynb 文件
nbconvert:
	jupyter nbconvert --to notebook --execute examples/workflow_by_code.ipynb

lint: black pylint flake8 mypy nbqa

########################################################################################
# 打包
########################################################################################

# 构建包。
build:
	python -m build --wheel

# 上传包。
upload:
	python -m twine upload dist/*

########################################################################################
# Documentation
########################################################################################

docs-gen:
	python -m sphinx.cmd.build -W docs $(PUBLIC_DIR)
