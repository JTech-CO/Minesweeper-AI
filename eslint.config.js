// Flat ESLint config. Enforces the package-boundary import rules from
// 파일트리 §2 / CLAUDE.md §5:
//   - @msai/core: zero deps, no ONNX, no DOM.
//   - @msai/agent <-> @msai/online-adapter: never import each other.
import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/.venv/**',
      '**/__pycache__/**',
      '**/.pytest_cache/**',
      '**/.ruff_cache/**',
      '**/*.config.js',
      '**/*.config.ts',
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // core has zero dependencies and must not reach for ONNX or other @msai packages.
    files: ['packages/core/**/*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            { name: 'onnxruntime-web', message: 'core must not depend on ONNX (파일트리 §2).' },
            { name: 'onnxruntime-node', message: 'core must not depend on ONNX (파일트리 §2).' },
          ],
          patterns: [
            { group: ['@msai/*'], message: 'core has zero deps; do not import other @msai packages.' },
          ],
        },
      ],
    },
  },
  {
    files: ['packages/agent/**/*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['@msai/online-adapter', '@msai/online-adapter/*'],
              message: 'agent must not import online-adapter (boundary, 파일트리 §2).',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['packages/online-adapter/**/*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['@msai/agent', '@msai/agent/*'],
              message: 'online-adapter must not import agent (boundary, 파일트리 §2).',
            },
          ],
        },
      ],
    },
  },
);
