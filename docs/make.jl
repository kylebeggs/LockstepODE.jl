using Documenter
using DocumenterVitepress
using LockstepODE

makedocs(;
    modules = [LockstepODE],
    authors = "Kyle Beggs",
    sitename = "LockstepODE.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/kylebeggs/LockstepODE.jl", # Update with your repo URL
        devbranch = "main",
        devurl = "dev"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md"
    ],
    warnonly = true
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/kylebeggs/LockstepODE.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main", # or master, trunk, ...
    push_preview = true
)
