import React from 'react'; // Import React for JSX

interface FrameProps {
  src: string;
  title?: string;
  width?: string | number;
  height?: string | number;
  allowFullScreen?: boolean;
}

export function Frame({ // Destructure props directly
  src,
  title = 'Embedded content',
  width = '100%',
  height = '450px',
  allowFullScreen = true,
}: FrameProps) {
  return (
    <div className="my-4"> {/* Wrapper for margin */}
      <iframe
        src={src}
        title={title}
        width={width}
        height={height}
        className="rounded border border-gray-300 dark:border-gray-700" // Tailwind classes for border and rounding
        allowFullScreen={allowFullScreen}
      />
    </div>
  );
} 