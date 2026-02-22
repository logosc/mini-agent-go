interface Props {
  errors: string[];
}

export function ErrorList({ errors }: Props) {
  if (errors.length === 0) return null;
  return (
    <div>
      {errors.map((err, i) => (
        <div key={i} className="error-box">
          {err}
        </div>
      ))}
    </div>
  );
}
