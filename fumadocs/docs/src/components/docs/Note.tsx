import React from 'react'; // Import React for JSX

export function Note({ children }: { children: React.ReactNode }) {
  return (
    <div 
      className="my-4 rounded border-l-4 border-blue-500 bg-blue-50 p-4 dark:border-blue-400 dark:bg-blue-900 dark:text-blue-100 text-blue-800"
      // Light mode: blue border, light blue background, dark blue text
      // Dark mode: lighter blue border, dark blue background, light blue text
    >
      {children}
    </div>
  );
} 