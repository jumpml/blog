import React from "react"
import { Link } from "gatsby"
import style from "./postlink.module.css"
import Img from "gatsby-image"
// import _ from "lodash"
// Component to place a conditional wrapper around content.
const ConditionalWrapper = ({ condition, wrapper, children }) =>
  condition ? wrapper(children) : <div>{children}</div>

const PostLink = ({ post }) => (
  <div>
    <li className={style.listitem}>
      {post.frontmatter.featimg && (
        <figure className={style.featimg}>
          <Link to={post.fields.slug}>
            <Img
              fixed={post.frontmatter.featimg.childImageSharp.fixed}
              alt={post.frontmatter.title}
            />
          </Link>
        </figure>
      )}

      <ConditionalWrapper
        // If featured image, wrap content in <div>.
        condition={post.frontmatter.featimg}
        wrapper={children => (
          <div className={style.article__wrap}>{children}</div>
        )}
      >
        <Link to={post.fields.slug}>
          <h1 className={style.article__title}>{post.frontmatter.title}</h1>
        </Link>

        <div className={style.article__meta}>
          by {post.frontmatter.author}. Published{" "}
          {new Date(post.frontmatter.date).toLocaleDateString("en-US", {
            month: "long",
            day: "numeric",
            year: "numeric",
          })}{" "}
        </div>
        {/* <div className={style.article__tax}>
          Filed under:{" "}
          {post.frontmatter.subject.map((subject, index) => [
            index > 0 && ", ",
            <Link key={index} to={`/subjects/${_.kebabCase(subject)}`}>
              {subject}
            </Link>,
          ])}
        </div> */}
        <div
          className={style.article__content}
          dangerouslySetInnerHTML={{ __html: post.excerpt }}
        />
      </ConditionalWrapper>
    </li>
  </div>
)
export default PostLink
