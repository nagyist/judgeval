import React from 'react';

interface TipProps {
  children: React.ReactNode;
}

export function Tip({ children }: TipProps) {
  return (
    <div 
      className="my-4 rounded border-l-4 border-green-500 bg-green-50 p-4 dark:border-green-400 dark:bg-green-900 dark:text-green-100 text-green-800"
    >
      {children}
    </div>
  );
} 