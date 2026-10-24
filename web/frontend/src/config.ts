import type { ConfigSchema, ConfigValues } from './types'

export function defaultsFor(schema: ConfigSchema): ConfigValues {
  return Object.fromEntries(
    Object.entries(schema).map(([name, field]) => [
      name,
      field.default ?? (field.type === 'bool' ? false : field.type === 'str' ? '' : 0),
    ]),
  )
}

/**
 * Form values from run values: the backend replaces an option with a preset
 * (resolvesTo) by the values it sets, so a resumed job shows the option again
 * and the preset's other fields go back to their defaults.
 */
export function formValuesFor(schema: ConfigSchema, values: ConfigValues): ConfigValues {
  const out = { ...values }
  for (const [name, field] of Object.entries(schema)) {
    for (const [option, preset] of Object.entries(field.resolvesTo ?? {})) {
      if (Object.entries(preset).every(([key, value]) => values[key] === value)) {
        out[name] = option
        for (const key of Object.keys(preset)) {
          if (key !== name && schema[key]?.default !== undefined) out[key] = schema[key].default!
        }
      }
    }
  }
  return out
}
