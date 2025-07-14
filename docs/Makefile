# Sphinx 文档的最小 makefile
#

# 你可以从命令行设置这些变量。
SPHINXOPTS    =
SPHINXBUILD   = python3 -msphinx
SPHINXPROJ    = Quantlab
SOURCEDIR     = .
BUILDDIR      = _build

# 把它放在最前面，这样没有参数的 "make" 就像 "make help" 一样。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 通用目标：将所有未知目标路由到 Sphinx，使用新的 "make mode" 选项。$(O) 作为 $(SPHINXOPTS) 的快捷方式。
%: Makefile
	pip install -r requirements.txt
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
