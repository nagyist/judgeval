import React from 'react';

interface WarningProps {
  children: React.ReactNode;
}

export function Warning({ children }: WarningProps) {
  return (
    <div 
      className="my-4 rounded border-l-4 border-red-500 bg-red-50 p-4 dark:border-red-400 dark:bg-red-900 dark:text-red-100 text-red-800"
      // Light mode: red border, light red background, dark red text
      // Dark mode: lighter red border, dark red background, light red text
    >
      {children}
    </div>
  );
} 