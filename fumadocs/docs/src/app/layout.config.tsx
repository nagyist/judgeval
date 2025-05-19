import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import React from 'react';

/**
 * Shared layout configurations
 *
 * you can customise layouts individually from:
 * Home Layout: app/(home)/layout.tsx
 * Docs Layout: app/docs/layout.tsx
 */
export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <div style={{ display: 'flex', alignItems: 'center' }}>
        {/* Light Mode Logo */}
        <span className="dark:hidden">
          <svg width="24" height="24" viewBox="0 0 527 542" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Judgment Logo Light" style={{ display: 'inline-block', verticalAlign: 'middle' }}>
            <path d="M418.534 541.154C403.442 541.154 264.88 541.154 264.88 541.154V492.304V443.453H391.078C391.078 443.453 437.213 443.453 437.213 400.299V153.802H526.912V400.299C526.912 446.517 517.6 481.547 498.975 505.39C480.35 529.233 453.536 541.154 418.534 541.154Z" fill="#0A0943"/>
            <rect x="424.381" width="102.534" height="102.534" rx="5" fill="#FD951F"/>
            <rect x="219.309" y="153.802" width="153.797" height="153.806" rx="5" fill="#1EEB96"/>
            <rect y="341.204" width="200.797" height="200.797" rx="5" fill="#5062FF"/>
          </svg>
        </span>
        {/* Dark Mode Logo */}
        <span className="hidden dark:inline">
          <svg width="24" height="24" viewBox="0 0 527 542" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Judgment Logo Dark" style={{ display: 'inline-block', verticalAlign: 'middle' }}>
            <path d="M418.534 541.154C406.557 541.154 291.066 541.057 243.906 541.017C232.867 541.008 224 532.056 224 521.017V488.5V463.453C224 452.407 232.954 443.453 244 443.453H391.078C391.078 443.453 391.078 443.453 391.078 443.453C391.078 443.453 437.213 443.453 437.213 400.299V173.802C437.213 162.756 446.167 153.802 457.213 153.802H506.912C517.958 153.802 526.912 162.756 526.912 173.802V400.299C526.912 446.517 517.6 481.547 498.975 505.39C480.35 529.233 453.536 541.154 418.534 541.154Z" fill="#F6F6F6"/>
            <rect y="342" width="200" height="200" rx="20" fill="#3E4EDA"/>
            <rect x="224" y="154" width="153" height="153" rx="20" fill="#34C49D"/>
            <rect x="438" y="27" width="89" height="89" rx="20" fill="#DB8D35"/>
          </svg>
        </span>
        <span style={{ marginLeft: '8px', verticalAlign: 'middle' }}>Judgment Labs Documentation</span>
      </div>
    ),
  },
  // see https://fumadocs.dev/docs/ui/navigation/links
  links: [],
};
