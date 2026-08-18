import type { ConfigSchema, ConfigValues } from './types'

export function defaultsFor(schema: ConfigSchema): ConfigValues {
  return Object.fromEntries(
    Object.entries(schema).map(([name, field]) => [
      name,
      field.default ?? (field.type === 'bool' ? false : field.type === 'str' ? '' : 0),
    ]),
  )
}
