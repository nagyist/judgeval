import React from 'react';
import Link from 'next/link'; // For internal navigation

interface CardProps {
  title: string;
  icon?: string; // We'll just display the name for now, or you can integrate an icon library
  href: string;
  children: React.ReactNode;
}

export function Card({ title, icon, href, children }: CardProps) {
  const isExternal = href.startsWith('http');
  const linkContent = (
    <div 
      className="block p-6 bg-card text-card-foreground rounded-lg border border-border shadow-md hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 transition-colors h-full"
    >
      {icon && <div className="mb-2 text-xl">{/* Icon would go here, e.g., <YourIconLibrary name={icon} /> */} {icon} </div>}
      <h5 className="mb-2 text-2xl font-bold tracking-tight">{title}</h5>
      <p className="font-normal text-muted-foreground text-sm">
        {children}
      </p>
    </div>
  );

  if (isExternal) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="no-underline">
        {linkContent}
      </a>
    );
  }

  return (
    <Link href={href} legacyBehavior passHref>
      <a className="no-underline">
        {linkContent}
      </a>
    </Link>
  );
} 