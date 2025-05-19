import React from 'react';

interface CardGroupProps {
  children: React.ReactNode;
  // You could add props for columns, e.g., cols?: number;
}

export function CardGroup({ children }: CardGroupProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 my-6">
      {children}
    </div>
  );
} 