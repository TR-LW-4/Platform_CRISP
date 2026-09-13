import type {
  ConfigSchema,
  ConfigValue,
  ConfigValues,
} from '../types'

interface ParameterFormProps {
  schema: ConfigSchema
  values: ConfigValues
  disabled?: boolean
  hideKeys?: string[]
  onChange: (values: ConfigValues) => void
}

export function ParameterForm({
  schema,
  values,
  disabled = false,
  hideKeys = [],
  onChange,
}: ParameterFormProps) {
  const setValue = (name: string, value: ConfigValue) => {
    onChange({ ...values, [name]: value })
  }
  const hidden = new Set(hideKeys)

  return (
    <div className="parameter-grid">
      {Object.entries(schema)
        .filter(([name, field]) => {
          if (hidden.has(name)) return false
          if (!field.visibleWhen) return true
          const conditions = Array.isArray(field.visibleWhen)
            ? field.visibleWhen
            : [field.visibleWhen]
          return conditions.every((condition) => {
            const controllingValue = values[condition.key]
            return condition.values.some(
              (candidate) => String(candidate) === String(controllingValue),
            )
          })
        })
        .map(([name, field]) => {
        const label = field.label ?? name.replaceAll('_', ' ')
        const options = field.options ?? field.choices
        const value = values[name] ?? field.default ?? ''

        return (
          <label className="field" key={name} title={field.help}>
            <span>
              {label}
              {field.help && <small className="help-dot">?</small>}
            </span>
            {field.type === 'bool' ? (
              <input
                type="checkbox"
                checked={Boolean(value)}
                disabled={disabled}
                onChange={(event) => setValue(name, event.target.checked)}
              />
            ) : options ? (
              <select
                value={String(value)}
                disabled={disabled}
                onChange={(event) => {
                  const selected = options.find(
                    (option) => String(option) === event.target.value,
                  )
                  setValue(name, selected ?? event.target.value)
                }}
              >
                {options.map((option) => (
                  <option key={String(option)} value={String(option)}>
                    {String(option)}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={field.type === 'str' ? 'text' : 'number'}
                value={String(value)}
                min={field.min}
                max={field.max}
                step={field.step ?? (field.type === 'float' ? 'any' : 1)}
                disabled={disabled}
                onChange={(event) => {
                  if (field.type === 'str') {
                    setValue(name, event.target.value)
                  } else {
                    setValue(
                      name,
                      field.type === 'int'
                        ? Number.parseInt(event.target.value || '0', 10)
                        : Number.parseFloat(event.target.value || '0'),
                    )
                  }
                }}
              />
            )}
          </label>
        )
      })}
    </div>
  )
}
