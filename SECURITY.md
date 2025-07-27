# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of our project seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to [security@roadvision-ai.com](mailto:security@roadvision-ai.com).

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

* **Type of issue** (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* **Full paths of source file(s) related to the vulnerability**
* **The location of the affected source code (tag/branch/commit or direct URL)**
* **Any special configuration required to reproduce the issue**
* **Step-by-step instructions to reproduce the issue**
* **Proof-of-concept or exploit code (if possible)**
* **Impact of the issue, including how an attacker might exploit it**

This information will help us triage your report more quickly.

## Preferred Languages

We prefer to receive vulnerability reports in English, but we can also accept reports in Russian.

## Policy

We will make a best effort to respond to security issues within 48 hours. We will keep you informed of our progress and any questions we may have.

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release new versions and update documentation

## Security Best Practices

When using this project, please follow these security best practices:

1. **Keep dependencies updated** - Regularly update all dependencies to their latest secure versions
2. **Use virtual environments** - Isolate project dependencies using virtual environments
3. **Validate input data** - Always validate and sanitize input data before processing
4. **Secure model files** - Keep model files secure and don't expose them publicly
5. **Monitor system resources** - Be aware of system resource usage during processing
6. **Use HTTPS** - When deploying, always use HTTPS for secure communication

## Security Features

This project includes several security features:

- **Input validation** - All input data is validated before processing
- **Error handling** - Comprehensive error handling prevents information leakage
- **Resource limits** - Built-in resource limits prevent DoS attacks
- **Secure defaults** - Secure default configurations
- **No hardcoded secrets** - No passwords or API keys in the codebase

## Contact

For security-related questions or concerns, please contact us at [security@roadvision-ai.com](mailto:security@roadvision-ai.com). 