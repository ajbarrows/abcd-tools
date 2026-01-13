# abcd-tools

A small collection of tools to make working with ABCD Study data a little bit easier.

Made with ❤️ by Tony Barrows (@ajbarrows).

## Get started for development

This project uses [Pixi](https://pixi.sh) for dependency management. First [install Pixi](https://pixi.sh/latest/#installation), then:

```bash
git clone git@github.com:ajbarrows/abcd-tools
cd abcd-tools
pixi install -e dev
pixi shell -e dev
```

### Available environments

- `default`: Core dependencies only
- `dev`: Development tools (testing, linting, docs)
- `modeling`: Adds PyMC and PyTorch for predictive modeling
- `matlab`: Adds MATLAB integration tools
- `full`: All features enabled

### Common tasks

```bash
# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Format code
pixi run format

# Lint code
pixi run lint

# Serve documentation locally
pixi run docs-serve
```
