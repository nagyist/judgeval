import React from 'react';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import { Accordion, Accordions} from 'fumadocs-ui/components/accordion';
import { Note } from './components/docs/Note';
import { Frame } from './components/docs/Frame';
import { Tip } from './components/docs/Tip';
import { Warning } from './components/docs/Warning';
import { TestComponent } from './components/docs/TestComponent';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { Card } from './components/docs/Card';
import { CardGroup } from './components/docs/CardGroup';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    Note,             // Register Note
    Frame,            // Register Frame
    Tip,              // Register Tip
    Warning,          // Register Warning
    TestComponent,    // Keep if still used
    Tabs,             // For explicit <Tabs> usage
    Tab, 
    Accordion, 
    Accordions,
    Card,
    CardGroup,
    // For explicit <Tab> usage
    pre: (props: React.HTMLAttributes<HTMLPreElement>) => {
      return (
        <CodeBlock {...props}>
          <Pre {...props} />
        </CodeBlock>
      );
    },
    ...components,
  };
}
