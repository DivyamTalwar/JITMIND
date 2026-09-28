// Narrow adaptation of Graft src/graph/extract.ts at 80692e5.
// Copyright (c) 2026 Context Graph Engine contributors. MIT: see LICENSE.
import Parser from 'tree-sitter';
import Python from 'tree-sitter-python';
export const PARSER = 'graft-python-subset/1;tree-sitter/0.21.1;python/0.21.0';
const parser = new Parser();
parser.setLanguage(Python);
const PY_KINDS = {class_definition: 'class', function_definition: 'function'};
const PARSE_CHUNK = 16384;

// Retains donor callback parsing, named-node traversal, scoped definitions,
// method promotion, header boundary, 1-based spans and bare callee extraction.
// Receiver bindings/imports/heritage are deliberately outside this subset.
export function extract(source) {
  const tree = parser.parse(index => source.slice(index, index + PARSE_CHUNK));
    if (tree.rootNode.hasError) return {status: 'parse_error', symbols: []};
    const symbols = [];
    let visits = 0;
    const walk = (node, scope, enclosingKind, owner) => {
      if (++visits > 50000 || scope.length > 64 || symbols.length > 1024) throw Error('parse_budget');
      const mapped = PY_KINDS[node.type];
      if (mapped) {
        const name = node.childForFieldName('name')?.text;
        if (name) {
          const kind = mapped === 'function' && enclosingKind === 'class' ? 'method' : mapped;
          const body = node.childForFieldName('body');
          const symbol = {name, qualified_name: [...scope, name].join('.'), kind,
            start_line: node.startPosition.row + 1, end_line: node.endPosition.row + 1,
            signature: source.slice(node.startIndex, body ? body.startIndex : node.endIndex).replace(/\s+/g, ' ').trim(),
            calls: []};
          symbols.push(symbol);
          for (const child of node.namedChildren) walk(child, [...scope, name], kind, symbol);
          return;
        }
      }
      if (node.type === 'call' && owner) {
        const fn = node.childForFieldName('function');
        if (fn?.type === 'identifier') {
          if (owner.calls.length >= 1024) throw Error('parse_budget');
          owner.calls.push(fn.text);
        }
      }
      for (const child of node.namedChildren) walk(child, scope, enclosingKind, owner);
    };
    walk(tree.rootNode, [], null, null);
    return {status: 'ok', symbols};
}
