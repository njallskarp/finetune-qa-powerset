[![Codacy Badge](https://app.codacy.com/project/badge/Grade/c588ec347bbc4f6988ef59694ed139c6)](https://app.codacy.com/gh/njallskarp/finetune-qa-powerset/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


# Finetune QA Powerset

## Testing

This project uses PyTest for unit testing, and Poetry for managing dependencies. Tests help ensure the integrity and reliability of the code, and they also serve as examples of how to use the various functions and classes. As you contribute, we strongly encourage you to add tests for your code.

### Installing Dependencies with Poetry

Before running tests, you'll need to install the project dependencies, including PyTest. This project uses Poetry, a Python packaging and dependency management tool. If you haven't installed Poetry yet, you can do so with this command:

```bash
curl -sSL https://install.python-poetry.org | python -
```

Once you've installed Poetry, navigate to the main project directory where the `pyproject.toml` and `poetry.lock` files are located. Install the project dependencies with:

```bash
poetry install
```

### Running Tests Locally

After installing dependencies, you can run the tests. Here's how:

1. **Activate the Poetry environment**:

 ```bash
 poetry shell
 ```

2. **Run the tests** with PyTest:

 ```bash
 pytest
 ```

 This will discover all the test files in the `tests` directory and run them.

### Writing Tests

When writing tests, here are some guidelines to follow:

1. **Place your test files in the `tests` directory.** PyTest will automatically discover these tests.

2. **Name your test file `test_*.py` or `*_test.py`.** This is the naming convention PyTest uses to discover test files.

3. **Write tests as functions named `test_*()`.** Again, this is the naming convention PyTest uses to discover test cases.

4. **Use assertions to verify outcomes.** For example, `assert func(10) == 42`.

5. **Consider normal cases, edge cases and error handling.** For example, test the system's behavior with normal sentences, empty strings, empty predictions, and so on. Ensure your code behaves as expected when given unusual or erroneous input.

### Continuous Integration

We also use GitHub Actions for Continuous Integration (CI). Whenever you open a pull request, the test suite will automatically run on GitHub's servers. You'll be able to see whether your changes passed all tests before the changes are reviewed. Please make sure your tests pass in this environment as well as locally.

This CI pipeline is defined in `.github/workflows/tests.yml`. If you need to modify the pipeline—for example, to add additional test environments or dependencies—please make sure to test the changes thoroughly.

### Additional Resources

Here are some additional resources on testing with PyTest, managing dependencies with Poetry, and PyTorch testing:

- [PyTest Documentation](https://docs.pytest.org/en/latest/)
- [Python Testing with PyTest (book)](https://pragprog.com/book/bopytest/python-testing-with-pytest)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [PyTorch Testing Documentation](https://pytorch.org/docs/stable/testing.html)

## Libraries Used and Rationale

Our project mainly uses the following libraries:

- **PyTorch**: We use PyTorch as the primary deep learning library because of its flexibility and efficiency. It provides a wide range of functionalities for building and training neural networks, as well as support for GPU acceleration.

- **Transformers (by Hugging Face)**: This library provides state-of-the-art general-purpose architectures (BERT, GPT-2, etc.) for Natural Language Understanding (NLU) and Natural Language Generation (NLG). It significantly simplifies the process of fine-tuning models on various linguistic tasks.

- **PyTest**: PyTest is a mature full-featured Python testing tool that helps us ensure the correctness and robustness of our code.

- **Poetry**: We use Poetry for dependency management because it's simple yet powerful. It allows us to declare our project’s libraries dependency in a clear and structured way.

These libraries are crucial to our project because they provide the tools we need to build, train, and test our models effectively and efficiently. They are all well-documented and widely used in the Python and Machine Learning communities, which means there's a wealth of knowledge and resources available for troubleshooting and learning.

## When to Test Your Code

Ideally, you should be testing your code at every step of the development process. Here are some specific times when you should run tests:

- **After writing a new function or method**: Write a test to confirm it behaves as expected.

- **After modifying existing code**: Run the relevant tests to ensure you haven't introduced any bugs.

- **Before pushing your changes**: Run the full test suite to catch any unforeseen issues.

- **Before opening a pull request**: Again, run the full test suite. Your PR will also be tested automatically via GitHub Actions.

## Testing Guidelines for Contributors

As a contributor, you play a crucial role in maintaining the quality and reliability of this project. Here are some guidelines to help you contribute effectively:

- **Write tests for your code**: Any new code should be accompanied by corresponding tests.

- **Follow the existing style**: To keep the codebase consistent and readable, please try to follow the same coding style and conventions used in the project.

- **Keep tests small and focused**: Each test should cover a single function or behavior. This makes it easier to identify the source of any failures.

- **Document your tests**: Use comments to explain what your tests are doing, especially if they're complex.

- **Run tests locally**: Before you push your changes or open a pull request, run the tests locally to catch any issues early.

Thank you for your contributions!