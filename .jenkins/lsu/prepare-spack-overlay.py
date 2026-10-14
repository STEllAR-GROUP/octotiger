#!/usr/bin/env python3
"""Copy site recipes into a job-local repository for Octotiger compatibility.

The HPX 1.9.1 backport for Clang, ROCm, and SYCL frontends comes from:
https://github.com/TheHPXProject/hpx/commit/ca5e2d0bb4e0546007dd2d21acbe302fd5d6a2cd
"""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import tokenize


def literal_string(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def package_class(recipe, name):
    classes = [node for node in ast.parse(recipe).body
               if isinstance(node, ast.ClassDef) and node.name == name]
    if len(classes) != 1:
        raise ValueError("Expected exactly one top-level " + name + " class in the site recipe")
    return classes[0]


def patch_hpx_recipe(recipe):
    cls = package_class(recipe, "Hpx")
    body = cls.body
    if ast.get_docstring(cls, clean=False) is not None:
        body = body[1:]
    if not body:
        raise ValueError("The site Hpx recipe has no package directives")
    patch_name = "hpx-1.9.1-clang-restricted-executor.patch"
    atomic_patch_name = "hpx-1.9.1-rocm-atomic-probe.patch"
    if patch_name in recipe or atomic_patch_name in recipe:
        raise ValueError("The source HPX recipe already contains this overlay patch")
    hooks = ("setup_build_environment", "setup_dependent_build_environment")
    if any(isinstance(node, ast.FunctionDef) and node.name in hooks for node in cls.body):
        raise ValueError("Unsupported existing HPX build-environment hook")
    lines = recipe.splitlines(keepends=True)
    insertion = body[0].lineno - 1
    indent = lines[insertion][:len(lines[insertion]) - len(lines[insertion].lstrip())]
    # The pipeline selects this modified recipe only for Clang-based frontends,
    # including ROCm and SYCL builds whose Spack compiler name can differ.
    lines.insert(insertion, indent + 'patch("' + patch_name + '", when="@1.9.1")\n'
                 + indent + 'patch("' + atomic_patch_name + '", when="@1.9.1 +rocm")\n')
    # HPXConfig also enables CUDA in consumers. Select native Clang there and
    # when building HPX, instead of letting CMake default to NVCC with Clang as host.
    environment_hooks = '''def setup_build_environment(self, env):
    super().setup_build_environment(env)
    if self.spec.satisfies("+cuda %clang"):
        env.set("CUDACXX", self.compiler.cxx)

def setup_dependent_build_environment(self, env, dependent_spec):
    super().setup_dependent_build_environment(env, dependent_spec)
    if self.spec.satisfies("+cuda %clang") and dependent_spec.satisfies("%clang"):
        env.set("CUDACXX", self.compiler.cxx)

'''
    lines.insert(insertion + 1, "".join(indent + line + "\n" if line else "\n"
                                      for line in environment_hooks.splitlines()))
    patched_recipe = "".join(lines)
    ast.parse(patched_recipe)
    return patched_recipe


def patch_octotiger_recipe(recipe):
    cls = package_class(recipe, "Octotiger")
    patch_name = "adapt-kokkos-for-hpx.patch"
    # Token edits retain the site's formatting and every unrelated directive.
    tokens = list(tokenize.generate_tokens(io.StringIO(recipe).readline))
    lines = recipe.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(position):
        return offsets[position[0] - 1] + position[1]

    edits = []
    references = [node for node in ast.walk(cls)
                  if literal_string(node) == patch_name]
    for call in ast.walk(cls):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "depends_on"):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        patches = keywords.get("patches")
        if not isinstance(patches, ast.List):
            continue
        matches = [node for node in patches.elts
                   if literal_string(node) == patch_name]
        if not matches:
            continue
        if len(matches) != 1 or not call.args:
            raise ValueError("Unsupported legacy Kokkos patch directive")
        # Recognize the public recipe's Kokkos string variable or a literal spec.
        dependency = call.args[0]
        if isinstance(dependency, ast.BinOp) and isinstance(dependency.op, ast.Add):
            dependency = dependency.left
        if isinstance(dependency, ast.Name) and dependency.id == "kokkos_string":
            definitions = [node.value for node in cls.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "kokkos_string"
                                   for target in node.targets)]
            if len(definitions) != 1:
                raise ValueError("Expected one literal kokkos_string definition")
            dependency = definitions[0]
        dependency_string = literal_string(dependency)
        is_kokkos = dependency_string and dependency_string.split()[0] == "kokkos"
        when = literal_string(keywords.get("when"))
        if not is_kokkos or not when or not all(part in when.split() for part in ("+kokkos", "+cuda", "%gcc")):
            raise ValueError("Legacy wrapper patch is not confined to GCC CUDA Kokkos")
        node = matches[0]
        indices = [i for i, token in enumerate(tokens)
                   if token.start == (node.lineno, node.col_offset)]
        if len(indices) != 1:
            raise ValueError("Cannot locate the legacy Kokkos patch token")
        index = indices[0]
        token = tokens[index]
        if token.type != tokenize.STRING or ast.literal_eval(token.string) != patch_name:
            raise ValueError("Unsupported legacy Kokkos patch string layout")
        start, end = offset(token.start), offset(token.end)
        if len(patches.elts) > 1:
            if patches.elts[-1] is node:
                comma = next(item for item in reversed(tokens[:index])
                             if item.type not in (tokenize.NL, tokenize.COMMENT))
                start = offset(comma.start)
            else:
                comma = next(item for item in tokens[index + 1:]
                             if item.type not in (tokenize.NL, tokenize.COMMENT))
                end = offset(comma.end)
            if comma.string != ",":
                raise ValueError("Unsupported legacy Kokkos patch list layout")
        edits.append((start, end, ""))
    if len(edits) != len(references):
        raise ValueError("Found an unsupported reference to the legacy Kokkos wrapper patch")

    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name == "cmake_args"]
    if len(methods) != 1:
        raise ValueError("Expected one Octotiger.cmake_args method")
    returns = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Return)]
    if (len(returns) != 1 or returns[0] is not methods[0].body[-1]
            or not isinstance(returns[0].value, ast.Name) or returns[0].value.id != "args"):
        raise ValueError("Expected Octotiger.cmake_args to end with a single return args")
    insertion = returns[0].lineno - 1
    indent = lines[insertion][:len(lines[insertion]) - len(lines[insertion].lstrip())]
    launcher = (indent + 'if self.spec.satisfies("+cuda +kokkos %gcc"):\n'
                + indent + '    args.append(self.define("CMAKE_CXX_COMPILER_LAUNCHER",\n'
                + indent + '        "python3;" + join_path(self.stage.source_path, ".jenkins", "lsu",\n'
                + indent + '                              "nvccHostDefinitions.py")\n'
                + indent + '        + ";--kokkos-compiler;" + join_path(self.spec["kokkos"].prefix,\n'
                + indent + '                                             "bin", "nvcc_wrapper")))\n'
                + indent + 'if self.spec.satisfies("%clang"):\n'
                + indent + '    args = [arg for arg in args if arg.split("=", 1)[0].split(":", 1)[0]\n'
                + indent + '            not in ("-DOCTOTIGER_WITH_BOOST_MULTIPRECISION",\n'
                + indent + '                    "-DOCTOTIGER_WITH_BLAST_TEST")]\n'
                + indent + '    args.append(self.define("OCTOTIGER_WITH_BOOST_MULTIPRECISION", True))\n'
                + indent + '    args.append(self.define("OCTOTIGER_WITH_BLAST_TEST", self.run_tests))\n')
    edits.append((offsets[insertion], offsets[insertion], launcher))
    for start, end, replacement in sorted(edits, reverse=True):
        recipe = recipe[:start] + replacement + recipe[end:]
    ast.parse(recipe)
    return recipe


def patch_kokkos_recipe(recipe, package_dir):
    cls = package_class(recipe, "Kokkos")
    lines = recipe.splitlines(keepends=True)
    remove_lines = []
    for patch_name in ("adapt-kokkos-for-nix.patch", "adapt-kokkos-for-hpx.patch"):
        matches = [node for node in cls.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Name) and node.value.func.id == "patch"
                   and node.value.args and literal_string(node.value.args[0]) == patch_name]
        references = [node for node in ast.walk(cls) if literal_string(node) == patch_name]
        if not matches and not references:
            continue
        if len(matches) != 1 or len(references) != 1:
            raise ValueError("Unsupported wrapper patch directive: " + patch_name)
        call = matches[0].value
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        allowed_conditions = (None, "@:4.1.00") if patch_name == "adapt-kokkos-for-nix.patch" else (None,)
        if (len(call.args) != 1 or set(keywords) - {"when"}
                or ("when" in keywords and literal_string(keywords["when"]) is None)
                or literal_string(keywords.get("when")) not in allowed_conditions):
            raise ValueError("Unexpected wrapper patch condition: " + patch_name)
        patch_lines = (package_dir / patch_name).read_text().splitlines()
        changes = [line for line in patch_lines
                   if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        files = [line for line in patch_lines if line.startswith("diff --git ")]
        if patch_name == "adapt-kokkos-for-nix.patch":
            shebangs = ["-#!/bin/bash -e", "+#!/usr/bin/env bash",
                        "-#!/bin/bash", "+#!/usr/bin/env bash"]
            old_cuda_root = ['-if [ ! -z $CUDA_ROOT ]; then',
                             '-  nvcc_compiler="$CUDA_ROOT/bin/nvcc"', '-fi']
            valid = changes in (shebangs, shebangs + old_cuda_root) and files == [
                "diff --git a/bin/kokkos_launch_compiler b/bin/kokkos_launch_compiler",
                "diff --git a/bin/nvcc_wrapper b/bin/nvcc_wrapper"]
        else:
            valid = changes == [
                '-  $host_command', '+  eval $host_command',
                '-    echo "TMPDIR=${temp_dir} $nvcc_command"',
                '+    echo "TMPDIR=${temp_dir} eval $nvcc_command"',
                '-  TMPDIR=${temp_dir} $nvcc_command',
                '+  TMPDIR=${temp_dir} eval $nvcc_command'] and files == [
                    "diff --git a/bin/nvcc_wrapper b/bin/nvcc_wrapper"]
        if not valid:
            raise ValueError("Unrecognized wrapper compatibility patch contents: " + patch_name)
        # Known directives occupy one source line; reject other layouts.
        line = lines[matches[0].lineno - 1]
        parsed = ast.parse(line.strip())
        if len(parsed.body) != 1 or ast.dump(parsed.body[0]) != ast.dump(matches[0]):
            raise ValueError("Unsupported wrapper patch source layout: " + patch_name)
        remove_lines.append(matches[0].lineno - 1)
    for index in sorted(remove_lines, reverse=True):
        del lines[index]
    result = "".join(lines)
    ast.parse(result)
    return result


def patch_ctest_timeout(recipe, seconds):
    cls = package_class(recipe, "Octotiger")
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name == "check"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one Octotiger check method")
    calls = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ctest"]
    if (len(calls) != 1 or len(calls[0].args) != 1 or calls[0].keywords
            or literal_string(calls[0].args[0]) != "--output-on-failure"):
        raise ValueError("Expected one ctest('--output-on-failure') call in Octotiger.check")
    lines = recipe.splitlines(keepends=True)
    index = calls[0].lineno - 1
    line = lines[index]
    try:
        parsed = ast.parse(line.strip())
    except SyntaxError as error:
        raise ValueError("Unsupported Octotiger ctest source layout") from error
    if (len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr)
            or ast.dump(parsed.body[0].value) != ast.dump(calls[0])):
        raise ValueError("Unsupported Octotiger ctest source layout")
    indent = line[:len(line) - len(line.lstrip())]
    # DPCPP 2023-03 installs sycl-ls; PI trace level 1 reports basic plugin loading.
    # Keep its environment and failures separate from the tests being diagnosed.
    diagnostic = '''if "+sycl ^dpcpp" in self.spec:
    import os
    import subprocess
    for name in ("CUDA_VISIBLE_DEVICES", "KOKKOS_VISIBLE_DEVICES",
                 "SYCL_DEVICE_FILTER", "ONEAPI_DEVICE_SELECTOR",
                 "SYCL_DEVICE_ALLOWLIST", "SYCL_PI_TRACE"):
        value = repr(os.environ[name]) if name in os.environ else "<unset>"
        print("SYCL test environment: " + name + "=" + value, flush=True)
    sycl_ls = os.path.join(str(self.spec["dpcpp"].prefix), "bin", "sycl-ls")
    if os.path.isfile(sycl_ls) and os.access(sycl_ls, os.X_OK):
        environment = os.environ.copy()
        environment["SYCL_PI_TRACE"] = "1"
        print("SYCL device diagnostic: " + sycl_ls + " (SYCL_PI_TRACE=1)", flush=True)
        try:
            result = subprocess.run([sycl_ls], env=environment, timeout=30, check=False)
            print("SYCL device diagnostic exit status: " + str(result.returncode), flush=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print("SYCL device diagnostic unavailable: " + str(error), flush=True)
    else:
        print("SYCL device diagnostic executable unavailable: " + sycl_ls, flush=True)
'''
    lines[index] = "".join(indent + part + "\n" for part in diagnostic.splitlines())
    lines[index] += indent + 'ctest("--output-on-failure", "--timeout", "' + str(seconds) + '")\n'
    result = "".join(lines)
    ast.parse(result)
    return result


def prepare(hpx_package_dir, octotiger_package_dir, kokkos_package_dir,
            copy_package_dirs, output_dir, ctest_timeout=None):
    if ctest_timeout is not None and ctest_timeout <= 0:
        raise ValueError("The CTest timeout must be a positive number of seconds")
    output_dir = output_dir.resolve()
    packages = []
    inputs = [("hpx", hpx_package_dir, patch_hpx_recipe),
              ("octotiger", octotiger_package_dir, patch_octotiger_recipe),
              ("kokkos", kokkos_package_dir,
               lambda recipe: patch_kokkos_recipe(recipe, kokkos_package_dir))]
    inputs += [(directory.name, directory, lambda recipe: recipe)
               for directory in copy_package_dirs]
    for name, package_dir, transform in inputs:
        if package_dir is None:
            continue
        if any(existing_name == name for existing_name, _, _ in packages):
            raise ValueError("Duplicate package requested: " + name)
        package_dir = package_dir.resolve()
        if output_dir == package_dir or package_dir in output_dir.parents:
            raise ValueError("The overlay must be outside the original package directory")
        recipe = transform((package_dir / "package.py").read_text())
        if name == "octotiger" and ctest_timeout is not None:
            recipe = patch_ctest_timeout(recipe, ctest_timeout)
        packages.append((name, package_dir, recipe))
    if not packages:
        raise ValueError("At least one source package directory is required")
    if ctest_timeout is not None and not any(name == "octotiger" for name, _, _ in packages):
        raise ValueError("A CTest timeout requires an Octotiger recipe")

    # Refuse to overwrite an existing scope. Never edit the original recipes.
    output_dir.mkdir(parents=True, exist_ok=False)
    repo = output_dir / "repo"
    manifest = {"packages": [], "misc_cache": str(output_dir / "cache"),
                "ctest_timeout_seconds": ctest_timeout}
    for name, package_dir, recipe in packages:
        target_package = repo / "packages" / name
        shutil.copytree(package_dir, target_package)
        (target_package / "package.py").write_text(recipe)
        original = (package_dir / "package.py").read_bytes()
        before_refs = {literal_string(node) for node in ast.walk(ast.parse(original))}
        after_refs = {literal_string(node) for node in ast.walk(ast.parse(recipe))}
        manifest["packages"].append({
            "name": name, "source": str(package_dir),
            "original_recipe_sha256": hashlib.sha256(original).hexdigest(),
            "overlay_recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
            "removed_patch_references": sorted(value for value in before_refs - after_refs
                                               if value and value.endswith(".patch")),
            "remaining_self_patch_directives": [
                {"patch": literal_string(node.args[0]),
                 "when": next((literal_string(keyword.value) for keyword in node.keywords
                                if keyword.arg == "when"), "always")}
                for node in ast.walk(ast.parse(recipe)) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "patch" and node.args]})
        if name == "hpx" and hpx_package_dir is not None:
            for patch_name in ("hpx-1.9.1-clang-restricted-executor.patch",
                               "hpx-1.9.1-rocm-atomic-probe.patch"):
                shutil.copyfile(Path(__file__).parent / "patches" / patch_name,
                                target_package / patch_name)
    (repo / "repo.yaml").write_text("repo:\n  namespace: octotiger_hpx191\n")
    config = output_dir / "config"
    config.mkdir()
    # A normal repos list prepends this repository while retaining lower scopes.
    (config / "repos.yaml").write_text("repos:\n- " + json.dumps(str(repo)) + "\n")
    # Spack keys package indexes by namespace, so parallel job overlays must not
    # share the user's index cache even when they share source/install caches.
    (config / "config.yaml").write_text("config:\n  misc_cache: "
                                        + json.dumps(str(output_dir / "cache")) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpx-package-dir", type=Path)
    parser.add_argument("--octotiger-package-dir", type=Path)
    parser.add_argument("--kokkos-package-dir", type=Path)
    parser.add_argument("--copy-package-dir", type=Path, action="append", default=[])
    parser.add_argument("--ctest-timeout", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.hpx_package_dir, args.octotiger_package_dir, args.kokkos_package_dir,
            args.copy_package_dir, args.output_dir, args.ctest_timeout)
